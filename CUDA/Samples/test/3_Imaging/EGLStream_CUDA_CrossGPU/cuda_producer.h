/*
 * eglvideoproducer.h
 *
 * Copyright (c) 2013-2014, NVIDIA CORPORATION. All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
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
#include <cuda_runtime.h>
#include <cuda.h>

typedef struct _test_cuda_producer_s
{
    //  Stream params
    CUcontext context;
    CUeglStreamConnection cudaConn;
    EGLStreamKHR eglStream;
    EGLDisplay eglDisplay;
    unsigned int charCnt;
    bool profileAPI;
    char *tempBuff;
    CUdeviceptr cudaPtr;
    CUdeviceptr cudaPtr1;
    CUstream prodCudaStream;
} test_cuda_producer_s;

CUresult cudaProducerInit(test_cuda_producer_s *cudaProducer, TestArgs *args);
CUresult cudaProducerPresentFrame(test_cuda_producer_s *parserArg, CUeglFrame cudaEgl, int t);
CUresult cudaProducerReturnFrame(test_cuda_producer_s *parserArg, CUeglFrame cudaEgl, int t);
CUresult cudaProducerDeinit(test_cuda_producer_s *cudaProducer);
CUresult cudaDeviceCreateProducer(test_cuda_producer_s *cudaProducer);
cudaError_t cudaProducer_filter(CUstream cStream, char *pSrc, int width, int height, char expectedVal, char newVal, int frameNumber);
void cudaProducerPrepareFrame(CUeglFrame *cudaEgl, CUdeviceptr cudaPtr, int bufferSize);
#endif

