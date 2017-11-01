/*
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
    NV12ToARGB color space conversion CUDA kernel

    This sample uses CUDA to perform a simple NV12 (YUV 4:2:0 planar)
    source and converts to output in ARGB format
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cudaProcessFrame.h"

__constant__ uint32 constAlpha;

#define MUL(x,y)    (x*y)
__constant__ float  constHueColorSpaceMat[9];


__device__ void YUV2RGB(uint32 *yuvi, float *red, float *green, float *blue)
{
    float luma, chromaCb, chromaCr;

    // Prepare for hue adjustment
    luma     = (float)yuvi[0];
    chromaCb = (float)((int32)yuvi[1] - 512.0f);
    chromaCr = (float)((int32)yuvi[2] - 512.0f);

    // Convert YUV To RGB with hue adjustment
    *red  = MUL(luma,     constHueColorSpaceMat[0]) +
            MUL(chromaCb, constHueColorSpaceMat[1]) +
            MUL(chromaCr, constHueColorSpaceMat[2]);
    *green= MUL(luma,     constHueColorSpaceMat[3]) +
            MUL(chromaCb, constHueColorSpaceMat[4]) +
            MUL(chromaCr, constHueColorSpaceMat[5]);
    *blue = MUL(luma,     constHueColorSpaceMat[6]) +
            MUL(chromaCb, constHueColorSpaceMat[7]) +
            MUL(chromaCr, constHueColorSpaceMat[8]);
}


__device__ uint32 RGBAPACK_8bit(float red, float green, float blue, uint32 alpha)
{
    uint32 ARGBpixel = 0;

    // Clamp final 10 bit results
    red   = min(max(red,   0.0f), 255.0f);
    green = min(max(green, 0.0f), 255.0f);
    blue  = min(max(blue,  0.0f), 255.0f);

    // Convert to 8 bit unsigned integers per color component
    ARGBpixel = (((uint32)blue) |
                 (((uint32)green) << 8)  |
                 (((uint32)red) << 16) | (uint32)alpha);

    return  ARGBpixel;
}

__device__ uint32 RGBAPACK_10bit(float red, float green, float blue, uint32 alpha)
{
    uint32 ARGBpixel = 0;

    // Clamp final 10 bit results
    red   = min(max(red,   0.0f), 1023.f);
    green = min(max(green, 0.0f), 1023.f);
    blue  = min(max(blue,  0.0f), 1023.f);

    // Convert to 8 bit unsigned integers per color component
    ARGBpixel = (((uint32)blue  >> 2) |
                 (((uint32)green >> 2) << 8)  |
                 (((uint32)red   >> 2) << 16) | (uint32)alpha);

    return  ARGBpixel;
}


// CUDA kernel for outputing the final ARGB output from NV12;
extern "C"
__global__ void Passthru_drvapi(uint32 *srcImage,   size_t nSourcePitch,
                                uint32 *dstImage,   size_t nDestPitch,
                                uint32 width,       uint32 height)
{
    int32 x, y;
    uint32 yuv101010Pel[2];
    uint32 processingPitch = ((width) + 63) & ~63;
    uint32 dstImagePitch   = nDestPitch >> 2;
    uint8 *srcImageU8     = (uint8 *)srcImage;

    processingPitch = nSourcePitch;

    // Pad borders with duplicate pixels, and we multiply by 2 because we process 2 pixels per thread
    x = blockIdx.x * (blockDim.x << 1) + (threadIdx.x << 1);
    y = blockIdx.y *  blockDim.y       +  threadIdx.y;

    if (x >= width)
        return; //x = width - 1;

    if (y >= height)
        return; // y = height - 1;

    // Read 2 Luma components at a time, so we don't waste processing since CbCr are decimated this way.
    // if we move to texture we could read 4 luminance values
    yuv101010Pel[0] = (srcImageU8[y * processingPitch + x    ]);
    yuv101010Pel[1] = (srcImageU8[y * processingPitch + x + 1]);

    // this steps performs the color conversion
    float luma[2];

    luma[0]   = (yuv101010Pel[0]        & 0x00FF);
    luma[1]   = (yuv101010Pel[1]        & 0x00FF);

    // Clamp the results to RGBA
    dstImage[y * dstImagePitch + x     ] = RGBAPACK_8bit(luma[0], luma[0], luma[0], constAlpha);
    dstImage[y * dstImagePitch + x + 1 ] = RGBAPACK_8bit(luma[1], luma[1], luma[1], constAlpha);
}


// CUDA kernel for outputing the final ARGB output from NV12;
extern "C"
__global__ void NV12ToARGB_drvapi(uint32 *srcImage,     size_t nSourcePitch,
                                  uint32 *dstImage,     size_t nDestPitch,
                                  uint32 width,         uint32 height)
{
    int32 x, y;
    uint32 yuv101010Pel[2];
    uint32 processingPitch = ((width) + 63) & ~63;
    uint32 dstImagePitch   = nDestPitch >> 2;
    uint8 *srcImageU8     = (uint8 *)srcImage;

    processingPitch = nSourcePitch;

    // Pad borders with duplicate pixels, and we multiply by 2 because we process 2 pixels per thread
    x = blockIdx.x * (blockDim.x << 1) + (threadIdx.x << 1);
    y = blockIdx.y *  blockDim.y       +  threadIdx.y;

    if (x >= width)
        return; //x = width - 1;

    if (y >= height)
        return; // y = height - 1;

    // Read 2 Luma components at a time, so we don't waste processing since CbCr are decimated this way.
    // if we move to texture we could read 4 luminance values
    yuv101010Pel[0] = (srcImageU8[y * processingPitch + x    ]) << 2;
    yuv101010Pel[1] = (srcImageU8[y * processingPitch + x + 1]) << 2;

    uint32 chromaOffset    = processingPitch * height;
    int32 y_chroma = y >> 1;

    if (y & 1)  // odd scanline ?
    {
        uint32 chromaCb;
        uint32 chromaCr;

        chromaCb = srcImageU8[chromaOffset + y_chroma * processingPitch + x    ];
        chromaCr = srcImageU8[chromaOffset + y_chroma * processingPitch + x + 1];

        if (y_chroma < ((height >> 1) - 1)) // interpolate chroma vertically
        {
            chromaCb = (chromaCb + srcImageU8[chromaOffset + (y_chroma + 1) * processingPitch + x    ] + 1) >> 1;
            chromaCr = (chromaCr + srcImageU8[chromaOffset + (y_chroma + 1) * processingPitch + x + 1] + 1) >> 1;
        }

        yuv101010Pel[0] |= (chromaCb << (COLOR_COMPONENT_BIT_SIZE       + 2));
        yuv101010Pel[0] |= (chromaCr << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));

        yuv101010Pel[1] |= (chromaCb << (COLOR_COMPONENT_BIT_SIZE       + 2));
        yuv101010Pel[1] |= (chromaCr << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));
    }
    else
    {
        yuv101010Pel[0] |= ((uint32)srcImageU8[chromaOffset + y_chroma * processingPitch + x    ] << (COLOR_COMPONENT_BIT_SIZE       + 2));
        yuv101010Pel[0] |= ((uint32)srcImageU8[chromaOffset + y_chroma * processingPitch + x + 1] << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));

        yuv101010Pel[1] |= ((uint32)srcImageU8[chromaOffset + y_chroma * processingPitch + x    ] << (COLOR_COMPONENT_BIT_SIZE       + 2));
        yuv101010Pel[1] |= ((uint32)srcImageU8[chromaOffset + y_chroma * processingPitch + x + 1] << ((COLOR_COMPONENT_BIT_SIZE << 1) + 2));
    }

    // this steps performs the color conversion
    uint32 yuvi[6];
    float red[2], green[2], blue[2];

    yuvi[0] = (yuv101010Pel[0] &   COLOR_COMPONENT_MASK);
    yuvi[1] = ((yuv101010Pel[0] >>  COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK);
    yuvi[2] = ((yuv101010Pel[0] >> (COLOR_COMPONENT_BIT_SIZE << 1)) & COLOR_COMPONENT_MASK);

    yuvi[3] = (yuv101010Pel[1] &   COLOR_COMPONENT_MASK);
    yuvi[4] = ((yuv101010Pel[1] >>  COLOR_COMPONENT_BIT_SIZE)       & COLOR_COMPONENT_MASK);
    yuvi[5] = ((yuv101010Pel[1] >> (COLOR_COMPONENT_BIT_SIZE << 1)) & COLOR_COMPONENT_MASK);

    // YUV to RGB Transformation conversion
    YUV2RGB(&yuvi[0], &red[0], &green[0], &blue[0]);
    YUV2RGB(&yuvi[3], &red[1], &green[1], &blue[1]);

    // Clamp the results to RGBA
    dstImage[y * dstImagePitch + x     ] = RGBAPACK_10bit(red[0], green[0], blue[0], constAlpha);
    dstImage[y * dstImagePitch + x + 1 ] = RGBAPACK_10bit(red[1], green[1], blue[1], constAlpha);
}

