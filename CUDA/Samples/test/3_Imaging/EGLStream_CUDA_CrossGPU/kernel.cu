/*
 * cuda_consumer.c
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
// DESCRIPTION:   Simple CUDA consumer rendering sample app
//

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include "eglstrm_common.h"

extern bool isCrossDevice;

__device__ static unsigned int numErrors = 0, errorFound = 0;
__device__ void checkProducerDataGPU(char *data, int size, char expectedVal, int frameNumber)
{
    if ((data[blockDim.x * blockIdx.x + threadIdx.x] != expectedVal) && (!errorFound)) {
        printf("Producer FOUND:%d expected: %d at %d for trial %d %d\n", data[blockDim.x * blockIdx.x + threadIdx.x], expectedVal, (blockDim.x * blockIdx.x + threadIdx.x), frameNumber, numErrors);
        numErrors++;
        errorFound = 1;
        return;
    }
}

__device__ void checkConsumerDataGPU(char *data, int size, char expectedVal, int frameNumber)
{
    if ((data[blockDim.x * blockIdx.x + threadIdx.x] != expectedVal) && (!errorFound)) {
        printf("Consumer FOUND:%d expected: %d at %d for trial %d %d\n", data[blockDim.x * blockIdx.x + threadIdx.x], expectedVal, (blockDim.x * blockIdx.x + threadIdx.x), frameNumber, numErrors);
        numErrors++;
        errorFound = 1;
        return;
    }
}

__global__ void writeDataToBuffer(char *pSrc, char newVal)
{
    pSrc[blockDim.x * blockIdx.x + threadIdx.x] = newVal;
}

__global__ void testKernelConsumer(char *pSrc, char size, char expectedVal, char newVal, int frameNumber)
{
    checkConsumerDataGPU(pSrc, size, expectedVal, frameNumber);
}


__global__ void testKernelProducer(char *pSrc, char size, char expectedVal, char newVal, int frameNumber)
{
    checkProducerDataGPU(pSrc, size, expectedVal, frameNumber);
}
__global__ void getNumErrors(int *numErr) {
        *numErr = numErrors;
}

extern "C" cudaError_t cudaProducer_filter(cudaStream_t pStream, char *pSrc, int width, int height, char expectedVal, char newVal, int frameNumber)
{
    // in case where consumer is on dgpu and producer is on igpu when return is called
    // the frame is not copied back to igpu. So the consumer changes is not visible to
    // producer
    if (isCrossDevice == 0) {
        testKernelProducer<<<(width * height)/1024, 1024, 1, pStream>>>(pSrc, width * height, expectedVal, newVal, frameNumber);
    }
    writeDataToBuffer<<<(width*height)/1024, 1024, 1, pStream>>>(pSrc, newVal);
    return cudaSuccess;
};

extern "C" cudaError_t cudaConsumer_filter(cudaStream_t cStream, char *pSrc, int width, int height, char expectedVal, char newVal, int frameNumber)
{
    testKernelConsumer<<<(width * height)/1024, 1024, 1, cStream>>>(pSrc, width * height, expectedVal, newVal, frameNumber);
    writeDataToBuffer<<<(width*height)/1024, 1024, 1, cStream>>>(pSrc, newVal);
    return cudaSuccess;
};

extern "C" cudaError_t cudaGetValueMismatch()
{
    int numErr_h;
    int *numErr_d = NULL;
    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&numErr_d, sizeof(int));
    if (err != cudaSuccess) {
        printf("Cuda Main: cudaMalloc failed with %s\n",cudaGetErrorString(err));
        return err;
    }
    getNumErrors<<<1,1>>>(numErr_d);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Cuda Main: cudaDeviceSynchronize failed with %s\n",cudaGetErrorString(err));
    }
    err = cudaMemcpy(&numErr_h, numErr_d, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Cuda Main: cudaMemcpy failed with %s\n",cudaGetErrorString(err));
        cudaFree(numErr_d);
        return err;
    }
    err = cudaFree(numErr_d);
    if (err != cudaSuccess) {
        printf("Cuda Main: cudaFree failed with %s\n",cudaGetErrorString(err));
        return err;
    }
    if (numErr_h > 0) {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}
