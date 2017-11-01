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


/* Simple example demonstrating how to use MPI with CUDA
*
*  Generate some random numbers on one node.
*  Dispatch them to all nodes.
*  Compute their square root on each node's GPU.
*  Compute the average of the results using MPI.
*
*  simpleMPI.cu: GPU part, compiled with nvcc
*/

#include <iostream>
using std::cerr;
using std::endl;

#include "simpleMPI.h"

// Error handling macro
#define CUDA_CHECK(call) \
    if((call) != cudaSuccess) { \
        cudaError_t err = cudaGetLastError(); \
        cerr << "CUDA error calling \""#call"\", code is " << err << endl; \
        my_abort(err); }


// Device code
// Very simple GPU Kernel that computes square roots of input numbers
__global__ void simpleMPIKernel(float *input, float *output)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    output[tid] = sqrt(input[tid]);
}


// Initialize an array with random data (between 0 and 1)
void initData(float *data, int dataSize)
{
    for (int i = 0; i < dataSize; i++)
    {
        data[i] = (float)rand() / RAND_MAX;
    }
}

// CUDA computation on each node
// No MPI here, only CUDA
void computeGPU(float *hostData, int blockSize, int gridSize)
{
    int dataSize = blockSize * gridSize;

    // Allocate data on GPU memory
    float *deviceInputData = NULL;
    CUDA_CHECK(cudaMalloc((void **)&deviceInputData, dataSize * sizeof(float)));

    float *deviceOutputData = NULL;
    CUDA_CHECK(cudaMalloc((void **)&deviceOutputData, dataSize * sizeof(float)));

    // Copy to GPU memory
    CUDA_CHECK(cudaMemcpy(deviceInputData, hostData, dataSize * sizeof(float), cudaMemcpyHostToDevice));

    // Run kernel
    simpleMPIKernel<<<gridSize, blockSize>>>(deviceInputData, deviceOutputData);

    // Copy data back to CPU memory
    CUDA_CHECK(cudaMemcpy(hostData, deviceOutputData, dataSize *sizeof(float), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(deviceInputData));
    CUDA_CHECK(cudaFree(deviceOutputData));
}

float sum(float *data, int size)
{
    float accum = 0.f;

    for (int i = 0; i < size; i++)
    {
        accum += data[i];
    }

    return accum;
}
