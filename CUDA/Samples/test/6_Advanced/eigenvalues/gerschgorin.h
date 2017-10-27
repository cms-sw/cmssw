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

#ifndef _GERSCHGORIN_H_
#define _GERSCHGORIN_H_

////////////////////////////////////////////////////////////////////////////////
//! Compute Gerschgorin interval for symmetric, tridiagonal matrix
//! @param  d  diagonal elements
//! @param  s  superdiagonal elements
//! @param  n  size of matrix
//! @param  lg  lower limit of Gerschgorin interval
//! @param  ug  upper limit of Gerschgorin interval
////////////////////////////////////////////////////////////////////////////////
extern "C" void
computeGerschgorin(float *d, float *s, unsigned int n, float &lg, float &ug);

#endif // #ifndef _GERSCHGORIN_H_
