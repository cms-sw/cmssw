/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

//! includes, system
#include <cmath>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cassert>
#include <fstream>
#include <algorithm>
#include <map>


// includes, projcet
#include "matlab.h"

// namespace, unnamed
namespace
{

} // end namespace, unnamed

///////////////////////////////////////////////////////////////////////////////
//! Write a tridiagonal, symmetric matrix in vector representation and
//! it's eigenvalues
//! @param  filename  name of output file
//! @param  d  diagonal entries of the matrix
//! @param  s  superdiagonal entries of the matrix (len = n - 1)
//! @param  eigenvals  eigenvalues of the matrix
//! @param  indices  vector of len n containing the position of the eigenvalues
//!                  if these are sorted in ascending order
//! @param  n  size of the matrix
///////////////////////////////////////////////////////////////////////////////
void
writeTridiagSymMatlab(const char *filename,
                      float *d, float *s,
                      float *eigenvals,
                      const unsigned int n)
{
    std::ofstream file(filename, std::ios::out);

    // write diagonal entries
    writeVectorMatlab(file, "d", d, n);

    // write superdiagonal entries
    writeVectorMatlab(file, "s", s, n-1);

    // write eigenvalues
    writeVectorMatlab(file, "eigvals", eigenvals, n);

    file.close();
}
