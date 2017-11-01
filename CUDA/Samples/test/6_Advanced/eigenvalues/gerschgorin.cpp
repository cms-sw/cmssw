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

/* Computation of Gerschgorin interval for symmetric, tridiagonal matrix */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>

#include "util.h"
#include "gerschgorin.h"

////////////////////////////////////////////////////////////////////////////////
//! Compute Gerschgorin interval for symmetric, tridiagonal matrix
//! @param  d  diagonal elements
//! @param  s  superdiagonal elements
//! @param  n  size of matrix
//! @param  lg  lower limit of Gerschgorin interval
//! @param  ug  upper limit of Gerschgorin interval
////////////////////////////////////////////////////////////////////////////////
void
computeGerschgorin(float *d, float *s, unsigned int n, float &lg, float &ug)
{

    lg = FLT_MAX;
    ug = -FLT_MAX;

    // compute bounds
    for (unsigned int i = 1; i < (n - 1); ++i)
    {

        // sum over the absolute values of all elements of row i
        float sum_abs_ni = fabsf(s[i-1]) + fabsf(s[i]);

        lg = min(lg, d[i] - sum_abs_ni);
        ug = max(ug, d[i] + sum_abs_ni);
    }

    // first and last row, only one superdiagonal element

    // first row
    lg = min(lg, d[0] - fabsf(s[0]));
    ug = max(ug, d[0] + fabsf(s[0]));

    // last row
    lg = min(lg, d[n-1] - fabsf(s[n-2]));
    ug = max(ug, d[n-1] + fabsf(s[n-2]));

    // increase interval to avoid side effects of fp arithmetic
    float bnorm = max(fabsf(ug), fabsf(lg));

    // these values depend on the implementation of floating count that is
    // employed in the following
    float psi_0 = 11 * FLT_EPSILON * bnorm;
    float psi_n = 11 * FLT_EPSILON * bnorm;

    lg = lg - bnorm * 2 * n * FLT_EPSILON - psi_0;
    ug = ug + bnorm * 2 * n * FLT_EPSILON + psi_n;

    ug = max(lg, ug);
}

