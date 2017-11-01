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

/* Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication using the CUDA driver API.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * CUBLAS provides high-performance matrix multiplication.
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 *
 * Volkov, V. 2010. Better performance at lower occupancy,
 * GPU Technology Conference 2~010 (GTC 2010).
 *
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <cstring>

#include <cuda.h>
#include <builtin_types.h>
#include <drvapi_error_string.h>

// includes, project
#include <helper_cuda_drvapi.h>
#include <helper_timer.h>
#include <helper_string.h>
#include <helper_image.h>

#include "matrixMul.h"

// includes, CUDA
const bool use_64bit_memory_address = false;

using namespace std;

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);
void randomInit(float *, int);

extern "C"
void computeGold(float *, const float *, const float *, unsigned int, unsigned int, unsigned int);

static CUresult initCUDA(int argc, char **argv, CUfunction *pMatrixMul);

//define input ptx file for different platforms
#if defined(_WIN64) || defined(__LP64__)
#define PTX_FILE "matrixMul_kernel64.ptx"
#define CUBIN_FILE "matrixMul_kernel64.cubin"
#else
#define PTX_FILE "matrixMul_kernel32.ptx"
#define CUBIN_FILE "matrixMul_kernel32.cubin"
#endif

////////////////////////////////////////////////////////////////////////////////
// Globals
////////////////////////////////////////////////////////////////////////////////
CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
size_t totalGlobalMem;

const char *sSDKsample = "matrixMulDrv (Driver API)";

void constantInit(float *data, int size, float val)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    printf("[ %s ]\n", sSDKsample);

    runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char **argv)
{
    // initialize CUDA
    CUfunction matrixMul = NULL;
    int block_size = 32;

    CUresult error_id = initCUDA(argc, argv, &matrixMul);

    if (error_id != CUDA_SUCCESS)
    {
        printf("initCUDA() returned %d\n-> %s\n", error_id, getCudaDrvErrorString(error_id));
        exit(EXIT_FAILURE);
    }

    // set seed for rand()
    srand(2006);

    // allocate host memory for matrices A and B
    unsigned int size_A = WA * HA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *) malloc(mem_size_A);
    unsigned int size_B = WB * HB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *) malloc(mem_size_B);

    // initialize host memory
    const float valB = 0.01f;
    constantInit(h_A, size_A, 1.0f);
    constantInit(h_B, size_B, valB);

    // First reserve about 4GB of memory, so we ensure that all memory allocated afterwards is > 4GB
    CUdeviceptr d_Mem[4];

    if (use_64bit_memory_address)
    {
        unsigned int mem_size = 1024*1024*1024;
        checkCudaErrors(cuMemAlloc(&d_Mem[0], mem_size));
        checkCudaErrors(cuMemAlloc(&d_Mem[1], mem_size));
        checkCudaErrors(cuMemAlloc(&d_Mem[2], mem_size));
        checkCudaErrors(cuMemAlloc(&d_Mem[3], mem_size));
    }

    // allocate device memory
    CUdeviceptr d_A;
    checkCudaErrors(cuMemAlloc(&d_A, mem_size_A));
    CUdeviceptr d_B;
    checkCudaErrors(cuMemAlloc(&d_B, mem_size_B));

    // copy host memory to device
    checkCudaErrors(cuMemcpyHtoD(d_A, h_A, mem_size_A));
    checkCudaErrors(cuMemcpyHtoD(d_B, h_B, mem_size_B));

    // allocate device memory for result
    size_t size_C = WC * HC;
    size_t mem_size_C = sizeof(float) * size_C;

    CUdeviceptr d_C;
    checkCudaErrors(cuMemAlloc(&d_C, mem_size_C));

    // allocate mem for the result on host side
    float *h_C = (float *) malloc(mem_size_C);

    // create and start timer
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);

    // start the timer
    sdkStartTimer(&timer);

    // There are two ways to launch CUDA kernels via the Driver API.
    // In this CUDA Sample, we illustrate both ways to pass parameters
    // and specify parameters.  By default we use the simpler method.
    dim3 block(block_size   , block_size   , 1);
    dim3 grid(WC/block_size, HC/block_size, 1);

    if (1)
    {
        // This is the new CUDA 4.0 API for Kernel Parameter passing and Kernel Launching (simplier method)
        if (use_64bit_memory_address && (totalGlobalMem > (unsigned long long)4*1024*1024*1024L))
        {
            size_t Matrix_Width_A = (size_t)WA;
            size_t Matrix_Width_B = (size_t)WB;
            void *args[5] = { &d_C, &d_A, &d_B, &Matrix_Width_A, &Matrix_Width_B};
            // new CUDA 4.0 Driver API Kernel launch call
            checkCudaErrors(cuLaunchKernel(matrixMul, grid.x, grid.y, grid.z,
                                           block.x, block.y, block.z,
                                           2*block_size*block_size*sizeof(float),
                                           NULL, args, NULL));

        }
        else
        {
            int Matrix_Width_A = WA;
            int Matrix_Width_B = WB;
            void *args[5] = { &d_C, &d_A, &d_B, &Matrix_Width_A, &Matrix_Width_B};
            // new CUDA 4.0 Driver API Kernel launch call
            checkCudaErrors(cuLaunchKernel(matrixMul, grid.x, grid.y, grid.z,
                                           block.x, block.y, block.z,
                                           2*block_size*block_size*sizeof(float),
                                           NULL, args, NULL));
        }

    }
    else
    {
        // This is the new CUDA 4.0 API for Kernel Parameter passing and Kernel Launching (advanced method)
        int offset = 0;
        char argBuffer[256];

        // pass in launch parameters (not actually de-referencing CUdeviceptr).  CUdeviceptr is
        // storing the value of the parameters
        *((CUdeviceptr *)&argBuffer[offset]) = d_C;
        offset += sizeof(d_C);
        *((CUdeviceptr *)&argBuffer[offset]) = d_A;
        offset += sizeof(d_A);
        *((CUdeviceptr *)&argBuffer[offset]) = d_B;
        offset += sizeof(d_B);

        if (use_64bit_memory_address && (totalGlobalMem > (unsigned long long)4*1024*1024*1024L))
        {
            size_t Matrix_Width_A = (size_t)WA;
            size_t Matrix_Width_B = (size_t)WB;

            *((CUdeviceptr *)&argBuffer[offset]) = Matrix_Width_A;
            offset += sizeof(Matrix_Width_A);
            *((CUdeviceptr *)&argBuffer[offset]) = Matrix_Width_B;
            offset += sizeof(Matrix_Width_B);
        }
        else
        {
            int Matrix_Width_A = WA;
            int Matrix_Width_B = WB;

            *((int *)&argBuffer[offset]) = Matrix_Width_A;
            offset += sizeof(Matrix_Width_A);
            *((int *)&argBuffer[offset]) = Matrix_Width_B;
            offset += sizeof(Matrix_Width_B);
        }

        void *kernel_launch_config[5] =
        {
            CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
            CU_LAUNCH_PARAM_BUFFER_SIZE,    &offset,
            CU_LAUNCH_PARAM_END
        };

        // new CUDA 4.0 Driver API Kernel launch call
        checkCudaErrors(cuLaunchKernel(matrixMul, grid.x, grid.y, grid.z,
                                       block.x, block.y, block.z,
                                       2*block_size*block_size*sizeof(float),
                                       NULL, NULL, (void **)&kernel_launch_config));
    }

    // copy result from device to host
    checkCudaErrors(cuMemcpyDtoH((void *) h_C, d_C, mem_size_C));

    // stop and destroy timer
    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    printf("Checking computed result for correctness: ");
    bool correct = true;

    for (int i = 0; i < (int)(WC * HC); i++)
    {
        if (fabs(h_C[i] - (WA * valB)) > 1e-5)
        {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > 1e-5\n", i, h_C[i], WA*valB);
            correct = false;
        }
    }

    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    printf("\nNOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.\n");

    // clean up memory
    if (use_64bit_memory_address)
    {
        cuMemFree(d_Mem[0]);
        cuMemFree(d_Mem[1]);
        cuMemFree(d_Mem[2]);
        cuMemFree(d_Mem[3]);
    }

    free(h_A);
    free(h_B);
    free(h_C);
    checkCudaErrors(cuMemFree(d_A));
    checkCudaErrors(cuMemFree(d_B));
    checkCudaErrors(cuMemFree(d_C));
    checkCudaErrors(cuCtxDestroy(cuContext));
}

// Allocates a matrix with random float entries.
void randomInit(float *data, int size)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = rand() / (float)RAND_MAX;
    }
}

bool inline
findModulePath(const char *module_file, string &module_path, char **argv, string &ptx_source)
{
    char *actual_path = sdkFindFilePath(module_file, argv[0]);

    if (actual_path)
    {
        module_path = actual_path;
    }
    else
    {
        printf("> findModulePath file not found: <%s> \n", module_file);
        return false;
    }

    if (module_path.empty())
    {
        printf("> findModulePath file not found: <%s> \n", module_file);
        return false;
    }
    else
    {
        printf("> findModulePath <%s>\n", module_path.c_str());

        if (module_path.rfind(".ptx") != string::npos)
        {
            FILE *fp = fopen(module_path.c_str(), "rb");
            fseek(fp, 0, SEEK_END);
            int file_size = ftell(fp);
            char *buf = new char[file_size+1];
            fseek(fp, 0, SEEK_SET);
            fread(buf, sizeof(char), file_size, fp);
            fclose(fp);
            buf[file_size] = '\0';
            ptx_source = buf;
            delete[] buf;
        }

        return true;
    }
}

static CUresult
initCUDA(int argc, char **argv, CUfunction *pMatrixMul)
{
    CUfunction cuFunction = 0;
    CUresult status;
    int major = 0, minor = 0;
    char deviceName[100];
    string module_path, ptx_source;

    cuDevice = findCudaDeviceDRV(argc, (const char **)argv);

    // get compute capabilities and the devicename
    checkCudaErrors(cuDeviceComputeCapability(&major, &minor, cuDevice));
    checkCudaErrors(cuDeviceGetName(deviceName, 256, cuDevice));
    printf("> GPU Device has SM %d.%d compute capability\n", major, minor);

    checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, cuDevice));
    printf("  Total amount of global memory:     %llu bytes\n", (unsigned long long)totalGlobalMem);
    printf("  64-bit Memory Address:             %s\n", (totalGlobalMem > (unsigned long long)4*1024*1024*1024L) ? "YES" : "NO");

    status = cuCtxCreate(&cuContext, 0, cuDevice);

    if (CUDA_SUCCESS != status)
    {
        goto Error;
    }

    // first search for the module path before we load the results
    if (!findModulePath(PTX_FILE, module_path, argv, ptx_source))
    {
        if (!findModulePath(CUBIN_FILE, module_path, argv, ptx_source))
        {
            printf("> findModulePath could not find <matrixMul_kernel> ptx or cubin\n");
            status = CUDA_ERROR_NOT_FOUND;
            goto Error;
        }
    }
    else
    {
        printf("> initCUDA loading module: <%s>\n", module_path.c_str());
    }

    if (module_path.rfind("ptx") != string::npos)
    {
        // in this branch we use compilation with parameters
        const unsigned int jitNumOptions = 3;
        CUjit_option *jitOptions = new CUjit_option[jitNumOptions];
        void **jitOptVals = new void *[jitNumOptions];

        // set up size of compilation log buffer
        jitOptions[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
        int jitLogBufferSize = 1024;
        jitOptVals[0] = (void *)(size_t)jitLogBufferSize;

        // set up pointer to the compilation log buffer
        jitOptions[1] = CU_JIT_INFO_LOG_BUFFER;
        char *jitLogBuffer = new char[jitLogBufferSize];
        jitOptVals[1] = jitLogBuffer;

        // set up pointer to set the Maximum # of registers for a particular kernel
        jitOptions[2] = CU_JIT_MAX_REGISTERS;
        int jitRegCount = 32;
        jitOptVals[2] = (void *)(size_t)jitRegCount;

        status = cuModuleLoadDataEx(&cuModule, ptx_source.c_str(), jitNumOptions, jitOptions, (void **)jitOptVals);

        printf("> PTX JIT log:\n%s\n", jitLogBuffer);
    }
    else
    {
        status = cuModuleLoad(&cuModule, module_path.c_str());
    }

    if (CUDA_SUCCESS != status)
    {
        goto Error;
    }

#if USE_64BIT_MEMORY_ADDRESS

    if (totalGlobalMem > (unsigned long long)4*1024*1024*1024L)
    {
        status = cuModuleGetFunction(&cuFunction, cuModule, "matrixMul_bs32_64bit");
    }
    else
#endif
    {
        status = cuModuleGetFunction(&cuFunction, cuModule, "matrixMul_bs32_32bit");
    }

    if (CUDA_SUCCESS != status)
    {
        goto Error;
    }

    *pMatrixMul = cuFunction;

    return CUDA_SUCCESS;
Error:
    cuCtxDestroy(cuContext);
    return status;
}


