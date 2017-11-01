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

#ifndef QUASIRANDOMGENERATOR_GPU_CUH
#define QUASIRANDOMGENERATOR_GPU_CUH

#include <nvrtc_helper.h>
#include "quasirandomGenerator_common.h"

//Fast integer multiplication
#define MUL(a, b) __umul24(a, b)

// Global variables for nvrtc outputs
char *ptx;
size_t ptxSize;
CUmodule module;

////////////////////////////////////////////////////////////////////////////////
// GPU code
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// Niederreiter quasirandom number generation kernel
////////////////////////////////////////////////////////////////////////////////

//Table initialization routine
void initTableGPU(unsigned int tableCPU[QRNG_DIMENSIONS][QRNG_RESOLUTION])
{
    CUdeviceptr c_Table;
    checkCudaErrors(cuModuleGetGlobal(&c_Table, NULL, module, "c_Table"));
    checkCudaErrors(cuMemcpyHtoD(c_Table, tableCPU, QRNG_DIMENSIONS * QRNG_RESOLUTION * sizeof(unsigned int)));
}

//Host-side interface
void quasirandomGeneratorGPU(CUdeviceptr d_Output, unsigned int seed, unsigned int N)
{
    dim3 threads(128, QRNG_DIMENSIONS);
    dim3 cudaGridSize(128, 1, 1);

    CUfunction kernel_addr;
    checkCudaErrors(cuModuleGetFunction(&kernel_addr, module, "quasirandomGeneratorKernel"));

    void *args[] = { (void *)&d_Output, (void *)&seed, (void *)&N };
    checkCudaErrors(cuLaunchKernel(kernel_addr,
                                            cudaGridSize.x, cudaGridSize.y, cudaGridSize.z, /* grid dim */
                                            threads.x, threads.y, threads.z, /* block dim */
                                            0,0, /* shared mem, stream */
                                            &args[0], /* arguments */
                                            0));

    checkCudaErrors(cuCtxSynchronize());
}

void inverseCNDgpu(CUdeviceptr d_Output, unsigned int N)
{
    dim3 threads(128, 1,1);
    dim3 cudaGridSize(128, 1, 1);

    CUfunction kernel_addr;
    checkCudaErrors(cuModuleGetFunction(&kernel_addr, module, "inverseCNDKernel"));

    void *args[] = { (void *)&d_Output,  (void *)&N };
    checkCudaErrors(cuLaunchKernel(kernel_addr,
                                            cudaGridSize.x, cudaGridSize.y, cudaGridSize.z, /* grid dim */
                                            threads.x, threads.y, threads.z, /* block dim */
                                            0,0, /* shared mem, stream */
                                            &args[0], /* arguments */
                                            0));

    checkCudaErrors(cuCtxSynchronize());
}

#endif

