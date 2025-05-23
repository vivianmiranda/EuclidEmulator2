/* emulator.h
*  ==========
*  This file is part of EuclidEmulator2
*  Copyright (c) 2020 Mischa Knabenhans
*
*  EuclidEmulator2 is free software: you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation, either version 3 of the License, or
*  (at your option) any later version.
*
*  EuclidEmulator2 is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef EMULATOR_H
#define EMULATOR_H

#ifndef PATH_TO_EE2_DATA_FILE
#define PATH_TO_EE2_DATA_FILE "./ee2_bindata.dat"
#endif

#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline2d.h>
#include <fstream>
#include <vector>
#include <memory>
#include <array>
#include "cosmo.h"
#include <armadillo>

using namespace std;

class EuclidEmulator{
  private:
    static constexpr int npcs = 14; //;
    static constexpr int nz = 101;  // number of redshifts in the training data
    static constexpr int nk = 613;  // number of k modes in training data
    static constexpr int n_coeffs[14] = {53, 53, 117, 117, 53, \
                                         117, 117, 117, 117, 521, \
                                         117, 1539, 173, 457};

    static constexpr int lmax = 16;
    static constexpr int nindices = 8;

    std::array<std::shared_ptr<gsl_spline2d>,npcs+1> logklogz2pc_spline; 

    /* Private data containers */
    double * pc[15];             // principal components
    double pc_weights[14];      // PCA weights
    double * pce_coeffs[14];     // PCE coefficients
    double * pce_multiindex[14]; // PCE multi-indices
    double * univ_legendre[8]; // univariate legendre polynomials
    double * pce_basisfuncs;    // multivariate legendre polynomials

    /* Private member functions */
    void read_in_ee2_data_file();
    void pc_2d_interp();
    void print_info();

  public:
    /* Public members */
    double kvec[613];
    double Bvec[101][613];

    /* Public member functions */
    EuclidEmulator();
    void compute_nlc(Cosmology csm, vector<double> redshift, int n_redshift);
    void write_nlc2file(const string &filename, vector<double> zvec, int n_redshift);
};

#endif
