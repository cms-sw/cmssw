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

#include "common.h"

///////////////////////////////////////////////////////////////////////////////
/// \brief add two vectors of size _count_
///
/// CUDA kernel
/// \param[in]  op1   term one
/// \param[in]  op2   term two
/// \param[in]  count vector size
/// \param[out] sum   result
///////////////////////////////////////////////////////////////////////////////
__global__
void AddKernel(const float *op1, const float *op2, int count, float *sum)
{
    const int pos = threadIdx.x + blockIdx.x * blockDim.x;

    if (pos >= count) return;

    sum[pos] = op1[pos] + op2[pos];
}

///////////////////////////////////////////////////////////////////////////////
/// \brief add two vectors of size _count_
/// \param[in]  op1   term one
/// \param[in]  op2   term two
/// \param[in]  count vector size
/// \param[out] sum   result
///////////////////////////////////////////////////////////////////////////////
static
void Add(const float *op1, const float *op2, int count, float *sum)
{
    dim3 threads(256);
    dim3 blocks(iDivUp(count, threads.x));

    AddKernel<<<blocks, threads>>>(op1, op2, count, sum);
}
