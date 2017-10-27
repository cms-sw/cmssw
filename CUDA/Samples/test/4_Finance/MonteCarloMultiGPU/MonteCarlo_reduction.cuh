/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef MONTECARLO_REDUCTION_CUH
#define MONTECARLO_REDUCTION_CUH

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////////////////////
// This function calculates total sum for each of the two input arrays.
// SUM_N must be power of two
// Unrolling provides a bit of a performance improvement for small
// to medium path counts.
////////////////////////////////////////////////////////////////////////////////

template<class T, int SUM_N, int blockSize>
__device__ void sumReduce(T *sum, T *sum2, cg::thread_block &cta, cg::thread_block_tile<32> &tile32, __TOptionValue *d_CallValue)
{
    const int VEC = 32;
    const int tid = cta.thread_rank();

    T beta  = sum[tid];
    T beta2 = sum2[tid];
    T temp, temp2;

    for (int i = VEC/2; i > 0; i>>=1)
    {
        if (tile32.thread_rank() < i)
        {
                temp      = sum[tid+i];
                temp2     = sum2[tid+i];
                beta     += temp;
                beta2    += temp2;
                sum[tid]  = beta;
                sum2[tid] = beta2;
        }
        cg::sync(tile32);
    }
    cg::sync(cta);

    if (tid == 0)
    {
        beta  = 0;
        beta2 = 0;
        for (int i = 0; i < blockDim.x; i += VEC) 
        {
            beta  += sum[i];
            beta2 += sum2[i];
        }
        __TOptionValue t  = {beta, beta2};
        *d_CallValue = t;
    }
    cg::sync(cta);
}


#endif
