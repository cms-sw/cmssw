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

#ifndef INTERVAL_H
#define INTERVAL_H

#define DEVICE 0
#define TYPE double
#define NUM_RUNS (100)

typedef TYPE T;

int const BLOCK_SIZE = 64;
int const GRID_SIZE = 1024;
int const THREADS = GRID_SIZE * BLOCK_SIZE;
int const DEPTH_RESULT = 128;

#define CHECKED_CALL(func)                                     \
    do {                                                       \
        cudaError_t err = (func);                              \
        if (err != cudaSuccess) {                              \
            printf("%s(%d): ERROR: %s returned %s (err#%d)\n", \
                   __FILE__, __LINE__, #func,                  \
                   cudaGetErrorString(err), err);              \
            exit(EXIT_FAILURE);                                          \
        }                                                      \
    } while (0)

#endif
