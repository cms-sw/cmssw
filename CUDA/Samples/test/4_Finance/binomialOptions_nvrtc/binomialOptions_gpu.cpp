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

////////////////////////////////////////////////////////////////////////////////
// Global types and parameters
////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <cmath>

#include <helper_cuda.h>
#include <nvrtc_helper.h>
#include <cuda_runtime.h>

#include "binomialOptions_common.h"

#include "common_gpu_header.h"
#include "realtype.h"


//Preprocessed input option data
typedef struct
{
    real S;
    real X;
    real vDt;
    real puByDf;
    real pdByDf;

} __TOptionData;


static bool moduleLoaded = false;
char *ptx, *kernel_file;
size_t ptxSize;
CUmodule module;

////////////////////////////////////////////////////////////////////////////////
// Host-side interface to GPU binomialOptions
////////////////////////////////////////////////////////////////////////////////

extern "C" void binomialOptionsGPU(
    real *callValue,
    TOptionData  *optionData,
    int optN,
    int argc,
    char **argv
)
{
    if (!moduleLoaded) {
      kernel_file = sdkFindFilePath("binomialOptions_kernel.cu", argv[0]);
      compileFileToPTX(kernel_file, argc, argv, &ptx, &ptxSize, 0);
      module = loadPTX(ptx, argc, argv);
      moduleLoaded = true;
    }

    __TOptionData h_OptionData[MAX_OPTIONS];

    for (int i = 0; i < optN; i++)
    {
        const real      T = optionData[i].T;
        const real      R = optionData[i].R;
        const real      V = optionData[i].V;

        const real     dt = T / (real)NUM_STEPS;
        const real    vDt = V * sqrt(dt);
        const real    rDt = R * dt;
        //Per-step interest and discount factors
        const real     If = exp(rDt);
        const real     Df = exp(-rDt);
        //Values and pseudoprobabilities of upward and downward moves
        const real      u = exp(vDt);
        const real      d = exp(-vDt);
        const real     pu = (If - d) / (u - d);
        const real     pd = (real)1.0 - pu;
        const real puByDf = pu * Df;
        const real pdByDf = pd * Df;

        h_OptionData[i].S      = (real)optionData[i].S;
        h_OptionData[i].X      = (real)optionData[i].X;
        h_OptionData[i].vDt    = (real)vDt;
        h_OptionData[i].puByDf = (real)puByDf;
        h_OptionData[i].pdByDf = (real)pdByDf;
    }

    CUfunction kernel_addr;
    checkCudaErrors(cuModuleGetFunction(&kernel_addr, module, "binomialOptionsKernel"));

    CUdeviceptr d_OptionData;
    checkCudaErrors(cuModuleGetGlobal(&d_OptionData, NULL, module, "d_OptionData"));
    checkCudaErrors(cuMemcpyHtoD(d_OptionData, h_OptionData, optN * sizeof(__TOptionData)));

    dim3 cudaBlockSize(128,1,1);
    dim3 cudaGridSize(optN, 1, 1);

    checkCudaErrors(cuLaunchKernel(kernel_addr,
                                            cudaGridSize.x, cudaGridSize.y, cudaGridSize.z, /* grid dim */
                                            cudaBlockSize.x, cudaBlockSize.y, cudaBlockSize.z, /* block dim */
                                            0,0, /* shared mem, stream */
                                            NULL, /* arguments */
                                            0));

    checkCudaErrors(cuCtxSynchronize());

    CUdeviceptr d_CallValue;
    checkCudaErrors(cuModuleGetGlobal(&d_CallValue, NULL, module, "d_CallValue"));
    checkCudaErrors(cuMemcpyDtoH(callValue, d_CallValue, optN *sizeof(real)));
}
