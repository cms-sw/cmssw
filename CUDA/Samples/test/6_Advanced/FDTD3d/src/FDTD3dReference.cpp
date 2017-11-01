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

#include "FDTD3dReference.h"

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdio.h>


void generateRandomData(float *data, const int dimx, const int dimy, const int dimz, const float lowerBound, const float upperBound)
{
    srand(0);

    for (int iz = 0 ; iz < dimz ; iz++)
    {
        for (int iy = 0 ; iy < dimy ; iy++)
        {
            for (int ix = 0 ; ix < dimx ; ix++)
            {
                *data = (float)(lowerBound + ((float)rand() / (float)RAND_MAX) * (upperBound - lowerBound));
                ++data;
            }
        }
    }
}

void generatePatternData(float *data, const int dimx, const int dimy, const int dimz, const float lowerBound, const float upperBound)
{
    for (int iz = 0 ; iz < dimz ; iz++)
    {
        for (int iy = 0 ; iy < dimy ; iy++)
        {
            for (int ix = 0 ; ix < dimx ; ix++)
            {
                *data = (float)(lowerBound + ((float)iz / (float)dimz) * (upperBound - lowerBound));
                ++data;
            }
        }
    }
}

bool fdtdReference(float *output, const float *input, const float *coeff, const int dimx, const int dimy, const int dimz, const int radius, const int timesteps)
{
    const int     outerDimx    = dimx + 2 * radius;
    const int     outerDimy    = dimy + 2 * radius;
    const int     outerDimz    = dimz + 2 * radius;
    const size_t  volumeSize   = outerDimx * outerDimy * outerDimz;
    const int     stride_y     = outerDimx;
    const int     stride_z     = stride_y * outerDimy;
    float        *intermediate = 0;
    const float  *bufsrc       = 0;
    float        *bufdst       = 0;
    float        *bufdstnext   = 0;

    // Allocate temporary buffer
    printf(" calloc intermediate\n");
    intermediate = (float *)calloc(volumeSize, sizeof(float));

    // Decide which buffer to use first (result should end up in output)
    if ((timesteps % 2) == 0)
    {
        bufsrc     = input;
        bufdst     = intermediate;
        bufdstnext = output;
    }
    else
    {
        bufsrc     = input;
        bufdst     = output;
        bufdstnext = intermediate;
    }

    // Run the FDTD (naive method)
    printf(" Host FDTD loop\n");

    for (int it = 0 ; it < timesteps ; it++)
    {
        printf("\tt = %d\n", it);
        const float *src = bufsrc;
        float *dst       = bufdst;

        for (int iz = -radius ; iz < dimz + radius ; iz++)
        {
            for (int iy = -radius ; iy < dimy + radius ; iy++)
            {
                for (int ix = -radius ; ix < dimx + radius ; ix++)
                {
                    if (ix >= 0 && ix < dimx && iy >= 0 && iy < dimy && iz >= 0 && iz < dimz)
                    {
                        float value = (*src) * coeff[0];

                        for (int ir = 1 ; ir <= radius ; ir++)
                        {
                            value += coeff[ir] * (*(src + ir) + *(src - ir));                       // horizontal
                            value += coeff[ir] * (*(src + ir * stride_y) + *(src - ir * stride_y)); // vertical
                            value += coeff[ir] * (*(src + ir * stride_z) + *(src - ir * stride_z)); // in front & behind
                        }

                        *dst = value;
                    }
                    else
                    {
                        *dst = *src;
                    }

                    ++dst;
                    ++src;
                }
            }
        }

        // Rotate buffers
        float *tmp = bufdst;
        bufdst     = bufdstnext;
        bufdstnext = tmp;
        bufsrc = (const float *)tmp;
    }

    printf("\n");

    if (intermediate)
        free(intermediate);

    return true;
}

bool compareData(const float *output, const float *reference, const int dimx, const int dimy, const int dimz, const int radius, const float tolerance)
{
    for (int iz = -radius ; iz < dimz + radius ; iz++)
    {
        for (int iy = -radius ; iy < dimy + radius ; iy++)
        {
            for (int ix = -radius ; ix < dimx + radius ; ix++)
            {
                if (ix >= 0 && ix < dimx && iy >= 0 && iy < dimy && iz >= 0 && iz < dimz)
                {
                    // Determine the absolute difference
                    float difference = fabs(*reference - *output);
                    float error;

                    // Determine the relative error
                    if (*reference != 0)
                        error = difference / *reference;
                    else
                        error = difference;

                    // Check the error is within the tolerance
                    if (error > tolerance)
                    {
                        printf("Data error at point (%d,%d,%d)\t%f instead of %f\n", ix, iy, iz, *output, *reference);
                        return false;
                    }
                }

                ++output;
                ++reference;
            }
        }
    }

    return true;
}
