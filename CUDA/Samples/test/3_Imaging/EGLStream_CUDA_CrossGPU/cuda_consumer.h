/*
 * cuda_consumer.h
 *
 * Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
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
#include <cuda_runtime.h>
#include <cuda.h>

typedef struct _test_cuda_consumer_s
{
    CUcontext context;
    CUeglStreamConnection cudaConn;
    EGLDisplay eglDisplay;
    EGLStreamKHR eglStream;
    unsigned int charCnt;
    char* cudaBuf;
    bool profileAPI;
    unsigned char *pCudaCopyMem;
    CUstream consCudaStream;
} test_cuda_consumer_s;

CUresult cuda_consumer_init(test_cuda_consumer_s *cudaConsumer, TestArgs *args);
CUresult cuda_consumer_Deinit(test_cuda_consumer_s *cudaConsumer);
CUresult cudaConsumerAcquireFrame(test_cuda_consumer_s *data, int frameNumber);
CUresult cudaConsumerReleaseFrame(test_cuda_consumer_s *data, int frameNumber);
CUresult cudaDeviceCreateConsumer(test_cuda_consumer_s *cudaConsumer);
cudaError_t cudaConsumer_filter(CUstream cStream, char *pSrc, int width, int height, char expectedVal, char newVal, int frameNumber);
cudaError_t cudaGetValueMismatch(void);

#endif

