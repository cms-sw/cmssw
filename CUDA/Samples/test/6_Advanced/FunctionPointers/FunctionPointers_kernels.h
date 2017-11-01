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

#ifndef __SOBELFILTER_KERNELS_H_
#define __SOBELFILTER_KERNELS_H_

typedef unsigned char Pixel;

// global determines which filter to invoke
enum SobelDisplayMode
{
    SOBELDISPLAY_IMAGE = 0,
    SOBELDISPLAY_SOBELTEX,
    SOBELDISPLAY_SOBELSHARED
};

// Enums to set up the function table
// note: if you change these be sure to recompile those files
// that include this header or ensure the .h is in the
// dependencies for the related object files
enum POINT_ENUM
{
    SOBEL_FILTER=0,
    BOX_FILTER,
    LAST_POINT_FILTER
};

enum BLOCK_ENUM
{
    THRESHOLD_FILTER = 0,
    NULL_FILTER,
    LAST_BLOCK_FILTER
};

extern enum SobelDisplayMode g_SobelDisplayMode;

extern "C" void sobelFilter(Pixel *odata, int iw, int ih, enum SobelDisplayMode mode, float fScale, int blockOperation, int pointOperation);
extern "C" void setupTexture(int iw, int ih, Pixel *data, int Bpp);
extern "C" void deleteTexture(void);
extern "C" void initFilter(void);
void setupFunctionTables();

#endif

