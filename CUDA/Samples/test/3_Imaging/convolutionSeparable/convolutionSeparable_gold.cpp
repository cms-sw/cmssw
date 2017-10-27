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


#include "convolutionSeparable_common.h"



////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
extern "C" void convolutionRowCPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR
)
{
    for (int y = 0; y < imageH; y++)
        for (int x = 0; x < imageW; x++)
        {
            float sum = 0;

            for (int k = -kernelR; k <= kernelR; k++)
            {
                int d = x + k;

                if (d >= 0 && d < imageW)
                    sum += h_Src[y * imageW + d] * h_Kernel[kernelR - k];
            }

            h_Dst[y * imageW + x] = sum;
        }
}



////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
extern "C" void convolutionColumnCPU(
    float *h_Dst,
    float *h_Src,
    float *h_Kernel,
    int imageW,
    int imageH,
    int kernelR
)
{
    for (int y = 0; y < imageH; y++)
        for (int x = 0; x < imageW; x++)
        {
            float sum = 0;

            for (int k = -kernelR; k <= kernelR; k++)
            {
                int d = y + k;

                if (d >= 0 && d < imageH)
                    sum += h_Src[d * imageW + x] * h_Kernel[kernelR - k];
            }

            h_Dst[y * imageW + x] = sum;
        }
}
