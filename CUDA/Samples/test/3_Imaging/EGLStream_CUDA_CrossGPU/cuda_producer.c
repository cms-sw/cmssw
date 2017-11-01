/**
 * cuda_producer.c
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
// DESCRIPTION:   Simple cuda EGL stream producer app
//

#include "cudaEGL.h"
#include "cuda_producer.h"
#include "eglstrm_common.h"
#include <cuda_runtime.h>
#if defined(EXTENSION_LIST)
EXTENSION_LIST(EXTLST_EXTERN)
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string.h>
#include "cuda_runtime.h"
#include "math.h"
int cudaPresentReturnData = INIT_DATA;
int fakePresent = 0;
CUeglFrame fakeFrame;
CUdeviceptr cudaPtrFake;
extern bool isCrossDevice;

void
cudaProducerPrepareFrame(CUeglFrame *cudaEgl, CUdeviceptr cudaPtr, int bufferSize)
{
    cudaEgl->frame.pPitch[0] = (void *)cudaPtr;
    cudaEgl->width = WIDTH;
    cudaEgl->depth = 0;
    cudaEgl->height = HEIGHT;
    cudaEgl->pitch = WIDTH * 4;
    cudaEgl->frameType = CU_EGL_FRAME_TYPE_PITCH;
    cudaEgl->planeCount = 1;
    cudaEgl->numChannels = 4;
    cudaEgl->eglColorFormat = CU_EGL_COLOR_FORMAT_ARGB;
    cudaEgl->cuFormat = CU_AD_FORMAT_UNSIGNED_INT8;
}


static int count_present = 0, count_return = 0;
static double present_time[25000] = { 0}, total_time_present = 0;
static double return_time[25000] = { 0}, total_time_return = 0;

void presentApiStat(void) ;
void presentApiStat(void) {
    int i = 0;
    double min = 10000000, max = 0;
    double average_launch_time = 0, standard_deviation = 0;
    if (count_present == 0) return;
    // lets compute the standard deviation
    min = max = present_time[1];
    average_launch_time = (total_time_present)/count_present;
    for(i = 1; i < count_present; i++) {
        standard_deviation += (present_time[i] - average_launch_time) * (present_time[i] - average_launch_time);
        if (present_time[i] < min)
            min = present_time[i];
        if (present_time[i] > max)
            max = present_time[i];
    }
    standard_deviation = sqrt(standard_deviation/count_present);
    printf("present Avg: %lf\n", average_launch_time);
    printf("present  SD: %lf\n", standard_deviation);
    printf("present min: %lf\n", min);
    printf("present max: %lf\n", max);

    min = max = return_time[1];
    average_launch_time = (total_time_return - return_time[0])/count_return;
    for(i = 1; i < count_return; i++) {
        standard_deviation += (return_time[i] - average_launch_time) * (return_time[i] - average_launch_time);
        if (return_time[i] < min)
            min = return_time[i];
        if (return_time[i] > max)
            max = return_time[i];
    }
    standard_deviation = sqrt(standard_deviation/count_return);
    printf("return  Avg: %lf\n", average_launch_time);
    printf("return   SD: %lf\n", standard_deviation);
    printf("return  min: %lf\n", min);
    printf("return  max: %lf\n", max);
}
CUresult
cudaProducerPresentFrame(test_cuda_producer_s *cudaProducer, CUeglFrame cudaEgl, int t)
{
    static int flag = 0;
    CUresult status = CUDA_SUCCESS;
    struct timespec start, end;
    double curTime;
    CUdeviceptr pDevPtr = (CUdeviceptr) cudaEgl.frame.pPitch[0];
    cudaProducer_filter(cudaProducer->prodCudaStream, (char*)pDevPtr, WIDTH*4, HEIGHT, cudaPresentReturnData, PROD_DATA+t, t);
    if (cudaProducer->profileAPI) {
        getTime(&start);
    }
    status = cuEGLStreamProducerPresentFrame(&cudaProducer->cudaConn, cudaEgl, &cudaProducer->prodCudaStream);
    if (status != CUDA_SUCCESS) {
        printf("Cuda Producer: Present frame failed, status:%d\n", status);
        goto done;
    }
    flag++;
    if (cudaProducer->profileAPI && flag > 10) {
        getTime(&end);
        curTime = TIME_DIFF(end, start);
        present_time[count_present++] = curTime;
        if (count_present == 25000) count_present = 0;
        total_time_present += curTime;
    }
done:
    return status;
}


int flag = 0;
CUresult
cudaProducerReturnFrame(test_cuda_producer_s *cudaProducer, CUeglFrame cudaEgl, int t)
{
    CUresult status = CUDA_SUCCESS;
    struct timespec start, end;
    double curTime;
    CUdeviceptr pDevPtr = 0;

    pDevPtr = (CUdeviceptr) cudaEgl.frame.pPitch[0];
    if (cudaProducer->profileAPI) {
        getTime(&start);
    }

    while(1)
    {        
        status = cuEGLStreamProducerReturnFrame(&cudaProducer->cudaConn, &cudaEgl, &cudaProducer->prodCudaStream);
        if (status == CUDA_ERROR_LAUNCH_TIMEOUT) {
            continue;
        } else
        if (status != CUDA_SUCCESS) {
            printf("Cuda Producer: Return frame failed, status:%d\n", status);
            goto done;
        }
        break;
    }
    if (cudaProducer->profileAPI) {
        getTime(&end);
        curTime = TIME_DIFF(end, start);
        return_time[count_return++] = curTime;
        if (count_return == 25000) count_return = 0;
        total_time_return += curTime;
    }
    if (flag % 2 == 0) {
        cudaPresentReturnData++;
    }
    cudaProducer_filter(cudaProducer->prodCudaStream, (char*)pDevPtr, WIDTH*4, HEIGHT, CONS_DATA+t, cudaPresentReturnData, t);
    flag++;
done:
    return status;
}

CUresult
cudaDeviceCreateProducer(test_cuda_producer_s *cudaProducer)
{
    CUdevice device;
    CUresult status = CUDA_SUCCESS;

    if (CUDA_SUCCESS != (status = cuInit(0))) {
        printf("Failed to initialize CUDA\n");
        return status;
    }

    if (CUDA_SUCCESS != (status = cuDeviceGet(&device, isCrossDevice ? 1 : 0))) {
        printf("failed to get CUDA device\n");
        return status;
    }

    if (CUDA_SUCCESS !=  (status = cuCtxCreate(&cudaProducer->context, 0, device)) ) {
        printf("failed to create CUDA context\n");
        return status;
    }
    cuCtxPopCurrent(&cudaProducer->context);
    return status;
}

CUresult
cudaProducerInit(test_cuda_producer_s *cudaProducer, TestArgs *args)
{
    CUresult status = CUDA_SUCCESS;
    int bufferSize;

    cudaProducer->charCnt = args->charCnt;
    bufferSize = cudaProducer->charCnt;

    cudaProducer->tempBuff = malloc(bufferSize);
    if (!cudaProducer->tempBuff) {
        printf("Cuda Producer: Failed to allocate image buffer\n");
        status = CUDA_ERROR_UNKNOWN;
        goto done;
    }
    memset((void*)cudaProducer->tempBuff, INIT_DATA, cudaProducer->charCnt);

    //Fill this init data
    status = cuMemAlloc(&cudaProducer->cudaPtr, bufferSize);
    if (status != CUDA_SUCCESS) {
        printf("Cuda Producer: cuda Malloc failed, status:%d\n", status);
        goto done;
    }
    status = cuMemcpyHtoD(cudaProducer->cudaPtr, (void*)(cudaProducer->tempBuff), bufferSize);
    if (status != CUDA_SUCCESS) {
        printf("Cuda Producer: cuMemCpy failed, status:%d\n", status);
        goto done;
    }

    //Fill this init data
    status = cuMemAlloc(&cudaProducer->cudaPtr1, bufferSize);
    if (status != CUDA_SUCCESS) {
        printf("Cuda Producer: cuda Malloc failed, status:%d\n", status);
        goto done;
    }
    status = cuMemcpyHtoD(cudaProducer->cudaPtr1, (void*)(cudaProducer->tempBuff), bufferSize);
    if (status != CUDA_SUCCESS) {
        printf("Cuda Producer: cuMemCpy failed, status:%d\n", status);
        goto done;
    }

    status = cuStreamCreate(&cudaProducer->prodCudaStream, 0);
    if (status != CUDA_SUCCESS) {
        printf("Cuda Producer: cuStreamCreate failed, status:%d\n", status);
        goto done;
    }

    //Fill this init data
    status = cuMemAlloc(&cudaPtrFake, 100);
    if (status != CUDA_SUCCESS) {
        printf("Cuda Producer: cuda Malloc failed, status:%d\n", status);
        goto done;
    }

    atexit(presentApiStat);
 done:
    return status;
}

CUresult
cudaProducerDeinit(test_cuda_producer_s *cudaProducer)
{
    if (cudaProducer->tempBuff) {
        free(cudaProducer->tempBuff);
    }
    if (cudaProducer->cudaPtr) {
        cuMemFree(cudaProducer->cudaPtr);
    }
    return cuEGLStreamProducerDisconnect(&cudaProducer->cudaConn);
}

