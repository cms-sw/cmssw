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
* \file DCT8x8_Gold.h
* \brief Contains declaration of CPU versions of DCT, IDCT and quantization
* routines.
*
* Contains declaration of CPU versions of DCT, IDCT and quantization
* routines.
*/


#pragma once

#include "BmpUtil.h"

extern "C"
{
    void computeDCT8x8Gold1(const float *fSrc, float *fDst, int Stride, ROI Size);
    void computeIDCT8x8Gold1(const float *fSrc, float *fDst, int Stride, ROI Size);
    void quantizeGoldFloat(float *fSrcDst, int Stride, ROI Size);
    void quantizeGoldShort(short *fSrcDst, int Stride, ROI Size);
    void computeDCT8x8Gold2(const float *fSrc, float *fDst, int Stride, ROI Size);
    void computeIDCT8x8Gold2(const float *fSrc, float *fDst, int Stride, ROI Size);
}
