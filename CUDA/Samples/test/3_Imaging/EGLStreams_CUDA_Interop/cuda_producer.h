/*
 * cuda_producer.h
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
// DESCRIPTION:   Simple cuda producer header file
//

#ifndef _CUDA_PRODUCER_H_
#define _CUDA_PRODUCER_H_
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include "cudaEGL.h"
#include "eglstrm_common.h"

extern EGLStreamKHR eglStream;
extern EGLDisplay   g_display; 

typedef struct _test_cuda_producer_s
{
    //  Stream params
    char *fileName1;
    char *fileName2;
    unsigned char *pBuff;
    int   frameCount;
    bool isARGB;
    bool pitchLinearOutput;
    unsigned int width;
    unsigned int height;
    CUcontext context;
    CUeglStreamConnection cudaConn;
    CUdeviceptr cudaPtrARGB[1];
    CUdeviceptr cudaPtrYUV[3];
    CUarray cudaArrARGB[1];
    CUarray cudaArrYUV[3];
    EGLStreamKHR eglStream;
    EGLDisplay eglDisplay;
} test_cuda_producer_s;

void cudaProducerInit(test_cuda_producer_s *cudaProducer, EGLDisplay eglDisplay, EGLStreamKHR eglStream, TestArgs *args);
CUresult cudaProducerTest(test_cuda_producer_s *parserArg, char *file);
CUresult cudaProducerDeinit(test_cuda_producer_s *cudaProducer);
CUresult cudaDeviceCreateProducer(test_cuda_producer_s *cudaProducer);
#endif

