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

#include "common_gpu_header.h"
#include "binomialOptions_common.h"
#include "realtype.h"

//Preprocessed input option data
typedef struct
{
    real S;
    real X;
    real vDt;
    real puByDf;
    real pdByDf;
} __TOptionData;
static __constant__ __TOptionData d_OptionData[MAX_OPTIONS];
__device__           real d_CallValue[MAX_OPTIONS];

#define THREADBLOCK_SIZE 128
#define ELEMS_PER_THREAD (NUM_STEPS/THREADBLOCK_SIZE)
#if NUM_STEPS % THREADBLOCK_SIZE
#error Bad constants
#endif


////////////////////////////////////////////////////////////////////////////////
// Overloaded shortcut functions for different precision modes
////////////////////////////////////////////////////////////////////////////////

#ifndef DOUBLE_PRECISION
__device__ inline float expiryCallValue(float S, float X, float vDt, int i)
{
    float d = S * __expf(vDt * (2.0f * i - NUM_STEPS)) - X;
    return (d > 0.0F) ? d : 0.0F;
}

#else
__device__ inline double expiryCallValue(double S, double X, double vDt, int i)
{
    double d = S * exp(vDt * (2.0 * i - NUM_STEPS)) - X;
    return (d > 0.0) ? d : 0.0;
}
#endif

////////////////////////////////////////////////////////////////////////////////
// GPU kernel
////////////////////////////////////////////////////////////////////////////////
extern "C" __global__ void binomialOptionsKernel()
{
    __shared__ real call_exchange[THREADBLOCK_SIZE + 1];

    const int     tid = threadIdx.x;
    const real      S = d_OptionData[blockIdx.x].S;
    const real      X = d_OptionData[blockIdx.x].X;
    const real    vDt = d_OptionData[blockIdx.x].vDt;
    const real puByDf = d_OptionData[blockIdx.x].puByDf;
    const real pdByDf = d_OptionData[blockIdx.x].pdByDf;

    real call[ELEMS_PER_THREAD + 1];
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i)
        call[i] = expiryCallValue(S, X, vDt, tid * ELEMS_PER_THREAD + i);

    if (tid == 0)
        call_exchange[THREADBLOCK_SIZE] = expiryCallValue(S, X, vDt, NUM_STEPS);

    int final_it = max(0, tid * ELEMS_PER_THREAD - 1);

    #pragma unroll 16
    for (int i = NUM_STEPS; i > 0; --i)
    {
        call_exchange[tid] = call[0];
        __syncthreads();
        call[ELEMS_PER_THREAD] = call_exchange[tid + 1];
        __syncthreads();

        if (i > final_it)
        {
           #pragma unroll
           for(int j = 0; j < ELEMS_PER_THREAD; ++j)
              call[j] = puByDf * call[j + 1] + pdByDf * call[j];
        }
    }

    if (tid == 0)
    {
        d_CallValue[blockIdx.x] = call[0];
    }
}
