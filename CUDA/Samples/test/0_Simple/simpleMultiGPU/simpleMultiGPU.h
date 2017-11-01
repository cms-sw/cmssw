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

/*
 * This application demonstrates how to use the CUDA API to use multiple GPUs.
 *
 * Note that in order to detect multiple GPUs in your system you have to disable
 * SLI in the nvidia control panel. Otherwise only one GPU is visible to the
 * application. On the other side, you can still extend your desktop to screens
 * attached to both GPUs.
 */

#ifndef SIMPLEMULTIGPU_H
#define SIMPLEMULTIGPU_H

typedef struct
{
    //Host-side input data
    int dataN;
    float *h_Data;

    //Partial sum for this GPU
    float *h_Sum;

    //Device buffers
    float *d_Data,*d_Sum;

    //Reduction copied back from GPU
    float *h_Sum_from_device;

    //Stream for asynchronous command execution
    cudaStream_t stream;

} TGPUplan;

extern "C"
void launch_reduceKernel(float *d_Result, float *d_Input, int N, int BLOCK_N, int THREAD_N, cudaStream_t &s);

#endif
