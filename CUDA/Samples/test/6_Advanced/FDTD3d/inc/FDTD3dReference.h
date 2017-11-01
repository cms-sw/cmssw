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

#ifndef _FDTD3DREFERENCE_H_
#define _FDTD3DREFERENCE_H_

void generateRandomData(float *data, const int dimx, const int dimy, const int dimz, const float lowerBound, const float upperBound);
void generatePatternData(float *data, const int dimx, const int dimy, const int dimz, const float lowerBound, const float upperBound);
bool fdtdReference(float *output, const float *input, const float *coeff, const int dimx, const int dimy, const int dimz, const int radius, const int timesteps);
bool compareData(const float *output, const float *reference, const int dimx, const int dimy, const int dimz, const int radius, const float tolerance=0.0001f);

#endif
