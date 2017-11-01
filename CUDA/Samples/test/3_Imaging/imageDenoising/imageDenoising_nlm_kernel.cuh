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
// NLM kernel
////////////////////////////////////////////////////////////////////////////////
__global__ void NLM(
    TColor *dst,
    int imageW,
    int imageH,
    float Noise,
    float lerpC
)
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    //Add half of a texel to always address exact texel centers
    const float x = (float)ix + 0.5f;
    const float y = (float)iy + 0.5f;

    if (ix < imageW && iy < imageH)
    {
        //Normalized counter for the NLM weight threshold
        float fCount = 0;
        //Total sum of pixel weights
        float sumWeights = 0;
        //Result accumulator
        float3 clr = {0, 0, 0};

        //Cycle through NLM window, surrounding (x, y) texel
        for (float i = -NLM_WINDOW_RADIUS; i <= NLM_WINDOW_RADIUS; i++)
            for (float j = -NLM_WINDOW_RADIUS; j <= NLM_WINDOW_RADIUS; j++)
            {
                //Find color distance from (x, y) to (x + j, y + i)
                float weightIJ = 0;

                for (float n = -NLM_BLOCK_RADIUS; n <= NLM_BLOCK_RADIUS; n++)
                    for (float m = -NLM_BLOCK_RADIUS; m <= NLM_BLOCK_RADIUS; m++)
                        weightIJ += vecLen(
                                        tex2D(texImage, x + j + m, y + i + n),
                                        tex2D(texImage,     x + m,     y + n)
                                    );

                //Derive final weight from color and geometric distance
                weightIJ     = __expf(-(weightIJ * Noise + (i * i + j * j) * INV_NLM_WINDOW_AREA));

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
void cuda_NLM(
    TColor *d_dst,
    int imageW,
    int imageH,
    float Noise,
    float lerpC
)
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

    NLM<<<grid, threads>>>(d_dst, imageW, imageH, Noise, lerpC);
}

////////////////////////////////////////////////////////////////////////////////
// Stripped NLM kernel, only highlighting areas with different LERP directions
////////////////////////////////////////////////////////////////////////////////
__global__ void NLMdiag(
    TColor *dst,
    unsigned int imageW,
    unsigned int imageH,
    float Noise,
    float lerpC
)
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    //Add half of a texel to always address exact texel centers
    const float x = (float)ix + 0.5f;
    const float y = (float)iy + 0.5f;

    if (ix < imageW && iy < imageH)
    {
        //Normalized counter for the weight threshold
        float fCount = 0;

        //Cycle through NLM window, surrounding (x, y) texel
        for (float i = -NLM_WINDOW_RADIUS; i <= NLM_WINDOW_RADIUS; i++)
            for (float j = -NLM_WINDOW_RADIUS; j <= NLM_WINDOW_RADIUS; j++)
            {

                //Find color distance between (x, y) and (x + j, y + i)
                float weightIJ = 0;

                for (float n = -NLM_BLOCK_RADIUS; n <= NLM_BLOCK_RADIUS; n++)
                    for (float m = -NLM_BLOCK_RADIUS; m <= NLM_BLOCK_RADIUS; m++)
                        weightIJ += vecLen(
                                        tex2D(texImage, x + j + m, y + i + n),
                                        tex2D(texImage,     x + m,     y + n)
                                    );

                //Derive final weight from color and geometric distance
                weightIJ = __expf(-(weightIJ * Noise + (i * i + j * j) * INV_NLM_WINDOW_AREA));

                //Increase the weight threshold counter
                fCount     += (weightIJ > NLM_WEIGHT_THRESHOLD) ? INV_NLM_WINDOW_AREA : 0;
            }

        //Choose LERP quotient basing on how many texels
        //within the NLM window exceeded the LERP threshold
        float lerpQ = (fCount > NLM_LERP_THRESHOLD) ? 1.0f : 0;

        //Write final result to global memory
        dst[imageW * iy + ix] = make_color(lerpQ, 0, (1.0f - lerpQ), 0);
    };
}

extern "C"
void cuda_NLMdiag(
    TColor *d_dst,
    int imageW,
    int imageH,
    float Noise,
    float lerpC
)
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

    NLMdiag<<<grid, threads>>>(d_dst, imageW, imageH, Noise, lerpC);
}
