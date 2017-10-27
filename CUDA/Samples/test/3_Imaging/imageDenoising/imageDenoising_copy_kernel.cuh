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



__global__ void Copy(
    TColor *dst,
    int imageW,
    int imageH
)
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    //Add half of a texel to always address exact texel centers
    const float x = (float)ix + 0.5f;
    const float y = (float)iy + 0.5f;

    if (ix < imageW && iy < imageH)
    {
        float4 fresult = tex2D(texImage, x, y);
        dst[imageW * iy + ix] = make_color(fresult.x, fresult.y, fresult.z, 0);
    }
}

extern "C" void
cuda_Copy(TColor *d_dst, int imageW, int imageH)
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));

    Copy<<<grid, threads>>>(d_dst, imageW, imageH);
}
