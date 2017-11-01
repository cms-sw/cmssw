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

/* Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 3
 * of the programming guide with some additions like error checking.
 *
 */

// Includes
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cstring>

// includes, project
#include <helper_functions.h>
#include <helper_cuda.h>

// includes, CUDA
#include <cuda.h>
#include <builtin_types.h>
#include <drvapi_error_string.h>

using namespace std;

// Variables
CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction vecAdd_kernel;
float *h_A;
float *h_B;
float *h_C;
CUdeviceptr d_A;
CUdeviceptr d_B;
CUdeviceptr d_C;
bool noprompt = false;

// Functions
void Cleanup(bool);
CUresult CleanupNoFailure();
void RandomInit(float *, int);
bool findModulePath(const char *, string &, char **, string &);
void ParseArguments(int, char **);

int *pArgc = NULL;
char **pArgv = NULL;

//define input ptx file for different platforms
#if defined(_WIN64) || defined(__LP64__)
#define PTX_FILE "vectorAdd_kernel64.ptx"
#else
#define PTX_FILE "vectorAdd_kernel32.ptx"
#endif

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(CUresult err, const char *file, const int line)
{
    if (CUDA_SUCCESS != err)
    {
        fprintf(stderr, "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, line %i.\n",
                err, getCudaDrvErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

inline int cudaDeviceInit(int ARGC, char **ARGV)
{
    int cuDevice = 0;
    int deviceCount = 0;
    CUresult err = cuInit(0);

    if (CUDA_SUCCESS == err)
    {
        checkCudaErrors(cuDeviceGetCount(&deviceCount));
    }

    if (deviceCount == 0)
    {
        fprintf(stderr, "cudaDeviceInit error: no devices supporting CUDA\n");
        exit(EXIT_FAILURE);
    }

    int dev = 0;
    dev = getCmdLineArgumentInt(ARGC, (const char **) ARGV, "device=");

    if (dev < 0)
    {
        dev = 0;
    }

    if (dev > deviceCount-1)
    {
        fprintf(stderr, "\n");
        fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
        fprintf(stderr, ">> cudaDeviceInit (-device=%d) is not a valid GPU device. <<\n", dev);
        fprintf(stderr, "\n");
        return -dev;
    }

    checkCudaErrors(cuDeviceGet(&cuDevice, dev));
    char name[100];
    cuDeviceGetName(name, 100, cuDevice);

    if (checkCmdLineFlag(ARGC, (const char **) ARGV, "quiet") == false)
    {
        printf("> Using CUDA Device [%d]: %s\n", dev, name);
    }

    return dev;
}

// This function returns the best GPU based on performance
inline int getMaxGflopsDeviceId()
{
    CUdevice current_device = 0, max_perf_device = 0;
    int device_count     = 0, sm_per_multiproc = 0;
    int max_compute_perf = 0, best_SM_arch     = 0;
    int major = 0, minor = 0, multiProcessorCount, clockRate;

    cuInit(0);
    checkCudaErrors(cuDeviceGetCount(&device_count));

    // Find the best major SM Architecture GPU device
    while (current_device < device_count)
    {
        checkCudaErrors(cuDeviceComputeCapability(&major, &minor, current_device));

        if (major > 0 && major < 9999)
        {
            best_SM_arch = MAX(best_SM_arch, major);
        }

        current_device++;
    }

    // Find the best CUDA capable GPU device
    current_device = 0;

    while (current_device < device_count)
    {
        checkCudaErrors(cuDeviceGetAttribute(&multiProcessorCount,
                                             CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                             current_device));
        checkCudaErrors(cuDeviceGetAttribute(&clockRate,
                                             CU_DEVICE_ATTRIBUTE_CLOCK_RATE,
                                             current_device));
        checkCudaErrors(cuDeviceComputeCapability(&major, &minor, current_device));

        if (major == 9999 && minor == 9999)
        {
            sm_per_multiproc = 1;
        }
        else
        {
            sm_per_multiproc = _ConvertSMVer2Cores(major, minor);
        }

        int compute_perf  = multiProcessorCount * sm_per_multiproc * clockRate;

        if (compute_perf  > max_compute_perf)
        {
            // If we find GPU with SM major > 2, search only these
            if (best_SM_arch > 2)
            {
                // If our device==dest_SM_arch, choose this, or else pass
                if (major == best_SM_arch)
                {
                    max_compute_perf  = compute_perf;
                    max_perf_device   = current_device;
                }
            }
            else
            {
                max_compute_perf  = compute_perf;
                max_perf_device   = current_device;
            }
        }

        ++current_device;
    }

    return max_perf_device;
}

// General initialization call to pick the best CUDA Device
inline CUdevice findCudaDevice(int argc, char **argv, int *p_devID)
{
    CUdevice cuDevice;
    int devID = 0;

    // If the command-line has a device number specified, use it
    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        devID = cudaDeviceInit(argc, argv);

        if (devID < 0)
        {
            printf("exiting...\n");
            exit(EXIT_SUCCESS);
        }
    }
    else
    {
        // Otherwise pick the device with highest Gflops/s
        char name[100];
        devID = getMaxGflopsDeviceId();
        checkCudaErrors(cuDeviceGet(&cuDevice, devID));
        cuDeviceGetName(name, 100, cuDevice);
        printf("> Using CUDA Device [%d]: %s\n", devID, name);
    }

    cuDeviceGet(&cuDevice, devID);

    if (p_devID)
    {
        *p_devID = devID;
    }

    return cuDevice;
}
// end of CUDA Helper Functions

// Host code
int main(int argc, char **argv)
{
    pArgc = &argc;
    pArgv = argv;

    printf("Vector Addition (Driver API)\n");
    int N = 50000, devID = 0;
    size_t  size = N * sizeof(float);

    CUresult error;
    ParseArguments(argc, argv);

    // Initialize
    checkCudaErrors(cuInit(0));

    // This assumes that the user is attempting to specify a explicit device -device=n
    if (argc > 1)
    {
        bool bFound = false;

        for (int param=0; param < argc; param++)
        {
            int string_start = 0;

            while (argv[param][string_start] == '-')
            {
                string_start++;
            }

            char *string_argv = &argv[param][string_start];

            if (!strncmp(string_argv, "device", 6))
            {
                int len=(int)strlen(string_argv);

                while (string_argv[len] != '=')
                {
                    len--;
                }

                devID = atoi(&string_argv[++len]);
                bFound = true;
            }

            if (bFound)
            {
                break;
            }
        }
    }

    // Get number of devices supporting CUDA
    int deviceCount = 0;
    error = cuDeviceGetCount(&deviceCount);

    if (error != CUDA_SUCCESS)
    {
        Cleanup(false);
    }

    if (deviceCount == 0)
    {
        printf("There is no device supporting CUDA.\n");
        Cleanup(false);
    }

    if (devID < 0)
    {
        devID = 0;
    }

    if (devID > deviceCount-1)
    {
        fprintf(stderr, "(Device=%d) invalid GPU device.  %d GPU device(s) detected.\nexiting...\n", devID, deviceCount);
        CleanupNoFailure();
        exit(EXIT_SUCCESS);
    }
    else
    {
        int major, minor;
        char deviceName[100];
        checkCudaErrors(cuDeviceComputeCapability(&major, &minor, devID));
        checkCudaErrors(cuDeviceGetName(deviceName, 256, devID));
        printf("> Using Device %d: \"%s\" with Compute %d.%d capability\n", devID, deviceName, major, minor);
    }

    // pick up device with zero ordinal (default, or devID)
    error = cuDeviceGet(&cuDevice, devID);

    if (error != CUDA_SUCCESS)
    {
        Cleanup(false);
    }

    // Create context
    error = cuCtxCreate(&cuContext, 0, cuDevice);

    if (error != CUDA_SUCCESS)
    {
        Cleanup(false);
    }

    // first search for the module path before we load the results
    string module_path, ptx_source;

    if (!findModulePath(PTX_FILE, module_path, argv, ptx_source))
    {
        if (!findModulePath("vectorAdd_kernel.cubin", module_path, argv, ptx_source))
        {
            printf("> findModulePath could not find <vectorAdd> ptx or cubin\n");
            Cleanup(false);
        }
    }
    else
    {
        printf("> initCUDA loading module: <%s>\n", module_path.c_str());
    }

    // Create module from binary file (PTX or CUBIN)
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

        error = cuModuleLoadDataEx(&cuModule, ptx_source.c_str(), jitNumOptions, jitOptions, (void **)jitOptVals);

        printf("> PTX JIT log:\n%s\n", jitLogBuffer);
    }
    else
    {
        error = cuModuleLoad(&cuModule, module_path.c_str());
    }

    if (error != CUDA_SUCCESS)
    {
        Cleanup(false);
    }

    // Get function handle from module
    error = cuModuleGetFunction(&vecAdd_kernel, cuModule, "VecAdd_kernel");

    if (error != CUDA_SUCCESS)
    {
        Cleanup(false);
    }

    // Allocate input vectors h_A and h_B in host memory
    h_A = (float *)malloc(size);

    if (h_A == 0)
    {
        Cleanup(false);
    }

    h_B = (float *)malloc(size);

    if (h_B == 0)
    {
        Cleanup(false);
    }

    h_C = (float *)malloc(size);

    if (h_C == 0)
    {
        Cleanup(false);
    }

    // Initialize input vectors
    RandomInit(h_A, N);
    RandomInit(h_B, N);

    // Allocate vectors in device memory
    error = cuMemAlloc(&d_A, size);

    if (error != CUDA_SUCCESS)
    {
        Cleanup(false);
    }

    error = cuMemAlloc(&d_B, size);

    if (error != CUDA_SUCCESS)
    {
        Cleanup(false);
    }

    error = cuMemAlloc(&d_C, size);

    if (error != CUDA_SUCCESS)
    {
        Cleanup(false);
    }

    // Copy vectors from host memory to device memory
    error = cuMemcpyHtoD(d_A, h_A, size);

    if (error != CUDA_SUCCESS)
    {
        Cleanup(false);
    }

    error = cuMemcpyHtoD(d_B, h_B, size);

    if (error != CUDA_SUCCESS)
    {
        Cleanup(false);
    }

#if 1

    if (1)
    {
        // This is the new CUDA 4.0 API for Kernel Parameter Passing and Kernel Launch (simpler method)

        // Grid/Block configuration
        int threadsPerBlock = 256;
        int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

        void *args[] = { &d_A, &d_B, &d_C, &N };

        // Launch the CUDA kernel
        error = cuLaunchKernel(vecAdd_kernel,  blocksPerGrid, 1, 1,
                               threadsPerBlock, 1, 1,
                               0,
                               NULL, args, NULL);

        if (error != CUDA_SUCCESS)
        {
            Cleanup(false);
        }
    }
    else
    {
        // This is the new CUDA 4.0 API for Kernel Parameter Passing and Kernel Launch (advanced method)
        int offset = 0;
        void *argBuffer[16];
        *((CUdeviceptr *)&argBuffer[offset]) = d_A;
        offset += sizeof(d_A);
        *((CUdeviceptr *)&argBuffer[offset]) = d_B;
        offset += sizeof(d_B);
        *((CUdeviceptr *)&argBuffer[offset]) = d_C;
        offset += sizeof(d_C);
        *((int *)&argBuffer[offset]) = N;
        offset += sizeof(N);

        // Grid/Block configuration
        int threadsPerBlock = 256;
        int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

        // Launch the CUDA kernel
        error = cuLaunchKernel(vecAdd_kernel,  blocksPerGrid, 1, 1,
                               threadsPerBlock, 1, 1,
                               0,
                               NULL, NULL, argBuffer);

        if (error != CUDA_SUCCESS)
        {
            Cleanup(false);
        }
    }

#else
    {
        char argBuffer[256];

        // pass in launch parameters (not actually de-referencing CUdeviceptr).  CUdeviceptr is
        // storing the value of the parameters
        *((CUdeviceptr *)&argBuffer[offset]) = d_A;
        offset += sizeof(d_A);
        *((CUdeviceptr *)&argBuffer[offset]) = d_B;
        offset += sizeof(d_B);
        *((CUdeviceptr *)&argBuffer[offset]) = d_C;
        offset += sizeof(d_C);
        *((int *)&argBuffer[offset]) = N;
        offset += sizeof(N);

        void *kernel_launch_config[5] =
        {
            CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
            CU_LAUNCH_PARAM_BUFFER_SIZE,    &offset,
            CU_LAUNCH_PARAM_END
        };

        // Grid/Block configuration
        int threadsPerBlock = 256;
        int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

        // Launch the CUDA kernel
        error = cuLaunchKernel(vecAdd_kernel,  blocksPerGrid, 1, 1,
                               threadsPerBlock, 1, 1,
                               0, 0,
                               NULL, (void **)&kernel_launch_config);

        if (error != CUDA_SUCCESS)
        {
            Cleanup(false);
        }
    }
#endif

#ifdef _DEBUG
    error = cuCtxSynchronize();

    if (error != CUDA_SUCCESS)
    {
        Cleanup(false);
    }

#endif

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    error = cuMemcpyDtoH(h_C, d_C, size);

    if (error != CUDA_SUCCESS)
    {
        Cleanup(false);
    }

    // Verify result
    int i;

    for (i = 0; i < N; ++i)
    {
        float sum = h_A[i] + h_B[i];

        if (fabs(h_C[i] - sum) > 1e-7f)
        {
            break;
        }
    }

    printf("%s\n", (i==N) ? "Result = PASS" : "Result = FAIL");

    exit((i==N) ? EXIT_SUCCESS : EXIT_FAILURE);
}

CUresult CleanupNoFailure()
{
    CUresult error;

    // Free device memory
    if (d_A)
    {
        error = cuMemFree(d_A);
    }

    if (d_B)
    {
        error = cuMemFree(d_B);
    }

    if (d_C)
    {
        error = cuMemFree(d_C);
    }

    // Free host memory
    if (h_A)
    {
        free(h_A);
    }

    if (h_B)
    {
        free(h_B);
    }

    if (h_C)
    {
        free(h_C);
    }

    error = cuCtxDestroy(cuContext);

    return error;
}

void Cleanup(bool noError)
{
    CUresult error;
    error = CleanupNoFailure();

    if (!noError || error != CUDA_SUCCESS)
    {
        printf("Function call failed\nFAILED\n");
        exit(EXIT_FAILURE);
    }

    if (!noprompt)
    {
        printf("\nPress ENTER to exit...\n");
        fflush(stdout);
        fflush(stderr);
        getchar();
    }
}


// Allocates an array with random float entries.
void RandomInit(float *data, int n)
{
    for (int i = 0; i < n; ++i)
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
        printf("> findModulePath could not find file: <%s> \n", module_file);
        return false;
    }
    else
    {
        printf("> findModulePath found file at <%s>\n", module_path.c_str());

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

// Parse program arguments
void ParseArguments(int argc, char **argv)
{
    for (int i = 0; i < argc; ++i)
    {
        if (strcmp(argv[i], "--noprompt") == 0 ||
            strcmp(argv[i], "-noprompt") == 0)
        {
            noprompt = true;
            break;
        }
    }
}
