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
* \file dct8x8_kernel1.cu
* \brief Contains 1st CUDA implementations of DCT, IDCT and quantization routines,
*        used in JPEG internal data processing. Device code.
*
* This code implements first CUDA versions of forward and inverse Discrete Cosine
* Transform to blocks of image pixels (of 8x8 size), as in JPEG standard. The data
* processing is done using floating point representation.
* The routine that performs quantization of coefficients can be found in
* dct8x8_kernel_quantization.cu file.
*/
#pragma once
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include "Common.h"


/**
*  This unitary matrix performs discrete cosine transform of rows of the matrix to the left
*/
__constant__ float DCTv8matrix[] =
{
    0.3535533905932738f,  0.4903926402016152f,  0.4619397662556434f,  0.4157348061512726f,  0.3535533905932738f,  0.2777851165098011f,  0.1913417161825449f,  0.0975451610080642f,
    0.3535533905932738f,  0.4157348061512726f,  0.1913417161825449f, -0.0975451610080641f, -0.3535533905932737f, -0.4903926402016152f, -0.4619397662556434f, -0.2777851165098011f,
    0.3535533905932738f,  0.2777851165098011f, -0.1913417161825449f, -0.4903926402016152f, -0.3535533905932738f,  0.0975451610080642f,  0.4619397662556433f,  0.4157348061512727f,
    0.3535533905932738f,  0.0975451610080642f, -0.4619397662556434f, -0.2777851165098011f,  0.3535533905932737f,  0.4157348061512727f, -0.1913417161825450f, -0.4903926402016153f,
    0.3535533905932738f, -0.0975451610080641f, -0.4619397662556434f,  0.2777851165098009f,  0.3535533905932738f, -0.4157348061512726f, -0.1913417161825453f,  0.4903926402016152f,
    0.3535533905932738f, -0.2777851165098010f, -0.1913417161825452f,  0.4903926402016153f, -0.3535533905932733f, -0.0975451610080649f,  0.4619397662556437f, -0.4157348061512720f,
    0.3535533905932738f, -0.4157348061512727f,  0.1913417161825450f,  0.0975451610080640f, -0.3535533905932736f,  0.4903926402016152f, -0.4619397662556435f,  0.2777851165098022f,
    0.3535533905932738f, -0.4903926402016152f,  0.4619397662556433f, -0.4157348061512721f,  0.3535533905932733f, -0.2777851165098008f,  0.1913417161825431f, -0.0975451610080625f
};


// Temporary blocks
__shared__ float CurBlockLocal1[BLOCK_SIZE2];
__shared__ float CurBlockLocal2[BLOCK_SIZE2];


/**
**************************************************************************
*  Performs 1st implementation of 8x8 block-wise Forward Discrete Cosine Transform of the given
*  image plane and outputs result to the array of coefficients.
*
* \param Dst            [OUT] - Coefficients plane
* \param ImgWidth       [IN] - Stride of Dst
* \param OffsetXBlocks  [IN] - Offset along X in blocks from which to perform processing
* \param OffsetYBlocks  [IN] - Offset along Y in blocks from which to perform processing
*
* \return None
*/
__global__ void CUDAkernel1DCT(float *Dst, int ImgWidth, int OffsetXBlocks, int OffsetYBlocks)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    // Block index
    const int bx = blockIdx.x + OffsetXBlocks;
    const int by = blockIdx.y + OffsetYBlocks;

    // Thread index (current coefficient)
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Texture coordinates
    const float tex_x = (float)((bx << BLOCK_SIZE_LOG2) + tx) + 0.5f;
    const float tex_y = (float)((by << BLOCK_SIZE_LOG2) + ty) + 0.5f;

    //copy current image pixel to the first block
    CurBlockLocal1[(ty << BLOCK_SIZE_LOG2) + tx ] = tex2D(TexSrc, tex_x, tex_y);

    //synchronize threads to make sure the block is copied
    cg::sync(cta);

    //calculate the multiplication of DCTv8matrixT * A and place it in the second block
    float curelem = 0;
    int DCTv8matrixIndex = 0 * BLOCK_SIZE + ty;
    int CurBlockLocal1Index = 0 * BLOCK_SIZE + tx;
#pragma unroll

    for (int i=0; i<BLOCK_SIZE; i++)
    {
        curelem += DCTv8matrix[DCTv8matrixIndex] * CurBlockLocal1[CurBlockLocal1Index];
        DCTv8matrixIndex += BLOCK_SIZE;
        CurBlockLocal1Index += BLOCK_SIZE;
    }

    CurBlockLocal2[(ty << BLOCK_SIZE_LOG2) + tx ] = curelem;

    //synchronize threads to make sure the first 2 matrices are multiplied and the result is stored in the second block
    cg::sync(cta);

    //calculate the multiplication of (DCTv8matrixT * A) * DCTv8matrix and place it in the first block
    curelem = 0;
    int CurBlockLocal2Index = (ty << BLOCK_SIZE_LOG2) + 0;
    DCTv8matrixIndex = 0 * BLOCK_SIZE + tx;
#pragma unroll

    for (int i=0; i<BLOCK_SIZE; i++)
    {
        curelem += CurBlockLocal2[CurBlockLocal2Index] * DCTv8matrix[DCTv8matrixIndex];
        CurBlockLocal2Index += 1;
        DCTv8matrixIndex += BLOCK_SIZE;
    }

    CurBlockLocal1[(ty << BLOCK_SIZE_LOG2) + tx ] = curelem;

    //synchronize threads to make sure the matrices are multiplied and the result is stored back in the first block
    cg::sync(cta);

    //copy current coefficient to its place in the result array
    Dst[ FMUL(((by << BLOCK_SIZE_LOG2) + ty), ImgWidth) + ((bx << BLOCK_SIZE_LOG2) + tx) ] = CurBlockLocal1[(ty << BLOCK_SIZE_LOG2) + tx ];
}


/**
**************************************************************************
*  Performs 1st implementation of 8x8 block-wise Inverse Discrete Cosine Transform of the given
*  DCT coefficients plane and outputs result to the image array
*
* \param Dst            [OUT] - Image plane
* \param ImgWidth       [IN] - Stride of Dst
* \param OffsetXBlocks  [IN] - Offset along X in blocks from which to perform processing
* \param OffsetYBlocks  [IN] - Offset along Y in blocks from which to perform processing
*
* \return None
*/
__global__ void CUDAkernel1IDCT(float *Dst, int ImgWidth, int OffsetXBlocks, int OffsetYBlocks)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    // Block index
    int bx = blockIdx.x + OffsetXBlocks;
    int by = blockIdx.y + OffsetYBlocks;

    // Thread index (current image pixel)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Texture coordinates
    const float tex_x = (float)((bx << BLOCK_SIZE_LOG2) + tx) + 0.5f;
    const float tex_y = (float)((by << BLOCK_SIZE_LOG2) + ty) + 0.5f;

    //copy current image pixel to the first block
    CurBlockLocal1[(ty << BLOCK_SIZE_LOG2) + tx ] = tex2D(TexSrc, tex_x, tex_y);

    //synchronize threads to make sure the block is copied
    cg::sync(cta);

    //calculate the multiplication of DCTv8matrix * A and place it in the second block
    float curelem = 0;
    int DCTv8matrixIndex = (ty << BLOCK_SIZE_LOG2) + 0;
    int CurBlockLocal1Index = 0 * BLOCK_SIZE + tx;
#pragma unroll

    for (int i=0; i<BLOCK_SIZE; i++)
    {
        curelem += DCTv8matrix[DCTv8matrixIndex] * CurBlockLocal1[CurBlockLocal1Index];
        DCTv8matrixIndex += 1;
        CurBlockLocal1Index += BLOCK_SIZE;
    }

    CurBlockLocal2[(ty << BLOCK_SIZE_LOG2) + tx ] = curelem;

    //synchronize threads to make sure the first 2 matrices are multiplied and the result is stored in the second block
    cg::sync(cta);

    //calculate the multiplication of (DCTv8matrix * A) * DCTv8matrixT and place it in the first block
    curelem = 0;
    int CurBlockLocal2Index = (ty << BLOCK_SIZE_LOG2) + 0;
    DCTv8matrixIndex = (tx << BLOCK_SIZE_LOG2) + 0;
#pragma unroll

    for (int i=0; i<BLOCK_SIZE; i++)
    {
        curelem += CurBlockLocal2[CurBlockLocal2Index] * DCTv8matrix[DCTv8matrixIndex];
        CurBlockLocal2Index += 1;
        DCTv8matrixIndex += 1;
    }

    CurBlockLocal1[(ty << BLOCK_SIZE_LOG2) + tx ] = curelem;

    //synchronize threads to make sure the matrices are multiplied and the result is stored back in the first block
    cg::sync(cta);

    //copy current coefficient to its place in the result array
    Dst[ FMUL(((by << BLOCK_SIZE_LOG2) + ty), ImgWidth) + ((bx << BLOCK_SIZE_LOG2) + tx) ] = CurBlockLocal1[(ty << BLOCK_SIZE_LOG2) + tx ];
}
