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



////////////////////////////////////////////////////////////////////////////////
//   0 1 2 3 4 5 6 7
// 0 + . . . . . . .
// 1 . . . . . . . .
// 2 . . . . . . . .
// 3 . . . * . . . .
// 4 . . . . . . . .
// 5 . . . . . . . .
// 6 . . . . . . . .
// 7 . . . . . . . .
//
// * - Base point for every thread, + - pixel around which ColorDistance is computed
// The idea behind this method:
// - Every thread in a 8x8 block computes just one ColorDistance
// - It is saved in the weights array that is shared across the threads
// - Threads are synced
// - For every pixel inside the block weights are considered to be constants
////////////////////////////////////////////////////////////////////////////////
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__global__ void NLM2(
    TColor *dst,
    int imageW,
    int imageH,
    float Noise,
    float lerpC
)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();

    //Weights cache
    __shared__ float fWeights[BLOCKDIM_X * BLOCKDIM_Y];

    const int   ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int   iy = blockDim.y * blockIdx.y + threadIdx.y;
    //Add half of a texel to always address exact texel centers
    const float  x = (float)ix  + 0.5f;
    const float  y = (float)iy  + 0.5f;
    const float cx = blockDim.x * blockIdx.x + NLM_WINDOW_RADIUS + 0.5f;
    const float cy = blockDim.x * blockIdx.y + NLM_WINDOW_RADIUS + 0.5f;

    if (ix < imageW && iy < imageH)
    {
        //Find color distance from current texel to the center of NLM window
        float weight = 0;

        for (float n = -NLM_BLOCK_RADIUS; n <= NLM_BLOCK_RADIUS; n++)
            for (float m = -NLM_BLOCK_RADIUS; m <= NLM_BLOCK_RADIUS; m++)
                weight += vecLen(
                              tex2D(texImage, cx + m, cy + n),
                              tex2D(texImage,  x + m,  y + n)
                          );

        //Geometric distance from current texel to the center of NLM window
        float dist =
            (threadIdx.x - NLM_WINDOW_RADIUS) * (threadIdx.x - NLM_WINDOW_RADIUS) +
            (threadIdx.y - NLM_WINDOW_RADIUS) * (threadIdx.y - NLM_WINDOW_RADIUS);

        //Derive final weight from color and geometric distance
        weight = __expf(-(weight * Noise + dist * INV_NLM_WINDOW_AREA));

        //Write the result to shared memory
        fWeights[threadIdx.y * BLOCKDIM_X + threadIdx.x] = weight;
        //Wait until all the weights are ready
        cg::sync(cta);


        //Normalized counter for the NLM weight threshold
        float fCount = 0;
        //Total sum of pixel weights
        float sumWeights = 0;
        //Result accumulator
        float3 clr = {0, 0, 0};

        int idx = 0;

        //Cycle through NLM window, surrounding (x, y) texel
        for (float i = -NLM_WINDOW_RADIUS; i <= NLM_WINDOW_RADIUS + 1; i++)
            for (float j = -NLM_WINDOW_RADIUS; j <= NLM_WINDOW_RADIUS + 1; j++)
            {
                //Load precomputed weight
                float weightIJ = fWeights[idx++];

                //Accumulate (x + j, y + i) texel color with computed weight
                float4 clrIJ = tex2D(texImage, x + j, y + i);
                clr.x       += clrIJ.x * weightIJ;
                clr.y       += clrIJ.y * weightIJ;
                clr.z       += clrIJ.z * weightIJ;

                //Sum of weights for color normalization to [0..1] range
                sumWeights  += weightIJ;

                //Update weight counter, if NLM weight for current window texel
                //exceeds the weight threshold
                fCount      += (weightIJ > NLM_WEIGHT_THRESHOLD) ? INV_NLM_WINDOW_AREA : 0;
            }

        //Normalize result color by sum of weights
        sumWeights = 1.0f / sumWeights;
        clr.x *= sumWeights;
        clr.y *= sumWeights;
        clr.z *= sumWeights;

        //Choose LERP quotient basing on how many texels
        //within the NLM window exceeded the weight threshold
        float lerpQ = (fCount > NLM_LERP_THRESHOLD) ? lerpC : 1.0f - lerpC;

        //Write final result to global memory
        float4 clr00 = tex2D(texImage, x, y);
        clr.x = lerpf(clr.x, clr00.x, lerpQ);
        clr.y = lerpf(clr.y, clr00.y, lerpQ);
        clr.z = lerpf(clr.z, clr00.z, lerpQ);
        dst[imageW * iy + ix] = make_color(clr.x, clr.y, clr.z, 0);
    }
}

extern "C"
void cuda_NLM2(
    TColor *d_dst,
    int imageW,
    int imageH,
    float Noise,
    float LerpC
)
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

    NLM2<<<grid, threads>>>(d_dst, imageW, imageH, Noise, LerpC);
}


////////////////////////////////////////////////////////////////////////////////
// Stripped NLM2 kernel, only highlighting areas with different LERP directions
////////////////////////////////////////////////////////////////////////////////
__global__ void NLM2diag(
    TColor *dst,
    int imageW,
    int imageH,
    float Noise,
    float LerpC
)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();

    //Weights cache
    __shared__ float fWeights[BLOCKDIM_X * BLOCKDIM_Y];

    const int   ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int   iy = blockDim.y * blockIdx.y + threadIdx.y;
    //Add half of a texel to always address exact texel centers
    const float  x = (float)ix  + 0.5f;
    const float  y = (float)iy  + 0.5f;
    const float cx = blockDim.x * blockIdx.x + NLM_WINDOW_RADIUS + 0.5f;
    const float cy = blockDim.x * blockIdx.y + NLM_WINDOW_RADIUS + 0.5f;

    if (ix < imageW && iy < imageH)
    {
        //Find color distance from current texel to the center of NLM window
        float weight = 0;

        for (float n = -NLM_BLOCK_RADIUS; n <= NLM_BLOCK_RADIUS + 1; n++)
            for (float m = -NLM_BLOCK_RADIUS; m <= NLM_BLOCK_RADIUS + 1; m++)
                weight += vecLen(
                              tex2D(texImage, cx + m, cy + n),
                              tex2D(texImage,  x + m,  y + n)
                          );

        //Geometric distance from current texel to the center of NLM window
        float dist =
            (threadIdx.x - NLM_WINDOW_RADIUS) * (threadIdx.x - NLM_WINDOW_RADIUS) +
            (threadIdx.y - NLM_WINDOW_RADIUS) * (threadIdx.y - NLM_WINDOW_RADIUS);

        //Derive final weight from color and geometric distance
        weight  = __expf(-(weight * Noise + dist * INV_NLM_WINDOW_AREA));

        //Write the result to shared memory
        fWeights[threadIdx.y * BLOCKDIM_X + threadIdx.x] = weight;
        //Wait until all the weights are ready
        cg::sync(cta);

        //Normalized counter for the NLM weight threshold
        float fCount = 0;
        int      idx = 0;

        //Cycle through NLM window, surrounding (x, y) texel
        for (float n = -NLM_WINDOW_RADIUS; n <= NLM_WINDOW_RADIUS + 1; n++)
            for (float m = -NLM_WINDOW_RADIUS; m <= NLM_WINDOW_RADIUS + 1; m++)
            {
                //Load precomputed weight
                float weightIJ = fWeights[idx++];

                //Update weight counter, if NLM weight for current window texel
                //exceeds the weight threshold
                fCount      += (weightIJ > NLM_WEIGHT_THRESHOLD) ? INV_NLM_WINDOW_AREA : 0;
            }

        //Choose LERP quotient basing on how many texels
        //within the NLM window exceeded the weight threshold
        float lerpQ = (fCount > NLM_LERP_THRESHOLD) ? 1.0f : 0.0f;

        //Write final result to global memory
        dst[imageW * iy + ix] = make_color(lerpQ, 0, (1.0f - lerpQ), 0);
    };
}

extern "C"
void cuda_NLM2diag(
    TColor *d_dst,
    int imageW,
    int imageH,
    float Noise,
    float LerpC
)
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

    NLM2diag<<<grid, threads>>>(d_dst, imageW, imageH, Noise, LerpC);
}
