/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */


///////////////////////////////////////////////////////////////////////////////
// Header for common includes and utility functions
///////////////////////////////////////////////////////////////////////////////

#ifndef COMMON_H
#define COMMON_H

///////////////////////////////////////////////////////////////////////////////
// Common includes
///////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <memory.h>
#include <math.h>

#include <helper_cuda.h>

///////////////////////////////////////////////////////////////////////////////
// Common constants
///////////////////////////////////////////////////////////////////////////////
const int StrideAlignment = 32;

///////////////////////////////////////////////////////////////////////////////
// Common functions
///////////////////////////////////////////////////////////////////////////////


// Align up n to the nearest multiple of m
inline int iAlignUp(int n, int m = StrideAlignment)
{
    int mod = n % m;

    if (mod)
        return n + m - mod;
    else
        return n;
}

// round up n/m
inline int iDivUp(int n, int m)
{
    return (n + m - 1) / m;
}

// swap two values
template<typename T>
inline void Swap(T &a, T &b)
{
    T t = a;
    a = b;
    b = t;
}
#endif
