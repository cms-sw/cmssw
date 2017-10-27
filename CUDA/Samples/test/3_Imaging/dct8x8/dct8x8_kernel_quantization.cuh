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

/**
**************************************************************************
* \file dct8x8_kernel_quantization.cu
* \brief Contains unoptimized quantization routines. Device code.
*
* This code implements CUDA versions of quantization of Discrete Cosine
* Transform coefficients with 8x8 blocks for float and short arrays.
*/

#pragma once
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include "Common.h"


/**
*  JPEG quality=0_of_12 quantization matrix
*/
__constant__ static short Q[] =
{
    32,  33,  51,  81,  66,  39,  34,  17,
    33,  36,  48,  47,  28,  23,  12,  12,
    51,  48,  47,  28,  23,  12,  12,  12,
    81,  47,  28,  23,  12,  12,  12,  12,
    66,  28,  23,  12,  12,  12,  12,  12,
    39,  23,  12,  12,  12,  12,  12,  12,
    34,  12,  12,  12,  12,  12,  12,  12,
    17,  12,  12,  12,  12,  12,  12,  12
};


/**
**************************************************************************
*  Performs in-place quantization of given DCT coefficients plane using
*  predefined quantization matrices (for floats plane). Unoptimized.
*
* \param SrcDst         [IN/OUT] - DCT coefficients plane
* \param Stride         [IN] - Stride of SrcDst
*
* \return None
*/
__global__ void CUDAkernelQuantizationFloat(float *SrcDst, int Stride)
{
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index (current coefficient)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //copy current coefficient to the local variable
    float curCoef = SrcDst[(by * BLOCK_SIZE + ty) * Stride + (bx * BLOCK_SIZE + tx) ];
    float curQuant = (float)Q[ ty * BLOCK_SIZE + tx ];

    //quantize the current coefficient
    float quantized = round(curCoef / curQuant);
    curCoef = quantized * curQuant;

    //copy quantized coefficient back to the DCT-plane
    SrcDst[(by * BLOCK_SIZE + ty) * Stride + (bx * BLOCK_SIZE + tx) ] = curCoef;
}


/**
**************************************************************************
*  Performs in-place quantization of given DCT coefficients plane using
*  predefined quantization matrices (for shorts plane). Unoptimized.
*
* \param SrcDst         [IN/OUT] - DCT coefficients plane
* \param Stride         [IN] - Stride of SrcDst
*
* \return None
*/
__global__ void CUDAkernelQuantizationShort(short *SrcDst, int Stride)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index (current coefficient)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //copy current coefficient to the local variable
    short curCoef = SrcDst[(by * BLOCK_SIZE + ty) * Stride + (bx * BLOCK_SIZE + tx) ];
    short curQuant = Q[ty * BLOCK_SIZE + tx];

    //quantize the current coefficient
    if (curCoef < 0)
    {
        curCoef = -curCoef;
        curCoef += curQuant>>1;
        curCoef /= curQuant;
        curCoef = -curCoef;
    }
    else
    {
        curCoef += curQuant>>1;
        curCoef /= curQuant;
    }

    cg::sync(cta);

    curCoef = curCoef * curQuant;

    //copy quantized coefficient back to the DCT-plane
    SrcDst[(by * BLOCK_SIZE + ty) * Stride + (bx * BLOCK_SIZE + tx) ] = curCoef;
}
