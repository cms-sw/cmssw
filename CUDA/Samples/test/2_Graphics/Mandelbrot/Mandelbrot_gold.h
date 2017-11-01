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

#ifndef _MANDELBROT_GOLD_h_
#define _MANDELBROT_GOLD_h_

#include <vector_types.h>

extern "C" void RunMandelbrotGold0(uchar4 *dst, const int imageW, const int imageH, const int crunch, const float xOff, const float yOff,
                                   const float xJParam, const float yJParam, const float scale, const uchar4 colors, const int frame, const int animationFrame, const bool isJulia);
extern "C" void RunMandelbrotDSGold0(uchar4 *dst, const int imageW, const int imageH, const int crunch, const double xOff, const double yOff,
                                     const double xJParam, const double yJParam, const double scale, const uchar4 colors, const int frame, const int animationFrame, const bool isJulia);
extern "C" void RunMandelbrotGold1(uchar4 *dst, const int imageW, const int imageH, const int crunch, const float xOff, const float yOff,
                                   const float xJParam, const float yJParam, const float scale, const uchar4 colors, const int frame, const int animationFrame, const bool isJulia);
extern "C" void RunMandelbrotDSGold1(uchar4 *dst, const int imageW, const int imageH, const int crunch, const double xOff, const double yOff,
                                     const double xJParam, const double yJParam, const double scale, const uchar4 colors, const int frame, const int animationFrame, const bool isJulia);

#endif
