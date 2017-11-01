/*
 * cuda_consumer.h
 *
 * Copyright 2016 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


//
// DESCRIPTION:   CUDA consumer header file
//

#ifndef _CUDA_CONSUMER_H_
#define _CUDA_CONSUMER_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cudaEGL.h"
#include "eglstrm_common.h"

extern EGLStreamKHR eglStream;
extern EGLDisplay   g_display; 

typedef struct _test_cuda_consumer_s
{
    CUcontext context;
    CUeglStreamConnection cudaConn;
    bool pitchLinearOutput;
    unsigned int width;
    unsigned int height;
    char *fileName1;
    char *fileName2;
    char *outFile1;
    char *outFile2;
    unsigned int frameCount;
} test_cuda_consumer_s;

void cuda_consumer_init(test_cuda_consumer_s *cudaConsumer, TestArgs *args);
CUresult cuda_consumer_deinit(test_cuda_consumer_s *cudaConsumer);
CUresult cudaConsumerTest (test_cuda_consumer_s *data, char *outFile);
CUresult cudaDeviceCreateConsumer(test_cuda_consumer_s *cudaConsumer);
#endif

