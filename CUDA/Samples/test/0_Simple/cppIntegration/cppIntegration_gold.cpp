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

/* Example of integrating CUDA functions into an existing
 * application / framework.
 * Reference solution computation.
 */

// Required header to support CUDA vector types
#include <vector_types.h>

////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C"
void computeGold(char *reference, char *idata, const unsigned int len);
extern "C"
void computeGold2(int2 *reference, int2 *idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! Each element is multiplied with the number of threads / array length
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
void
computeGold(char *reference, char *idata, const unsigned int len)
{
    for (unsigned int i = 0; i < len; ++i)
        reference[i] = idata[i] - 10;
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set for int2 version
//! Each element is multiplied with the number of threads / array length
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
void
computeGold2(int2 *reference, int2 *idata, const unsigned int len)
{
    for (unsigned int i = 0; i < len; ++i)
    {
        reference[i].x = idata[i].x - idata[i].y;
        reference[i].y = idata[i].y;
    }
}

