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
* This sample demonstrates how use texture fetches in CUDA
*
* This sample takes an input PGM image (image_filename) and generates
* an output PGM image (image_filename_out).  This CUDA kernel performs
* a simple 2D transform (rotation) on the texture coordinates (u,v).
* The results between simpleTexture and simpleTextureDrv are identical.
* The main difference is the implementation.  simpleTextureDrv makes calls
* to the CUDA driver API and demonstrates how to use cuModuleLoad to load
* the CUDA ptx (*.ptx) kernel just prior to kernel launch.
*
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
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

const char *image_filename = "lena_bw.pgm";
const char *ref_filename   = "ref_rotated.pgm";
float angle = 0.5f;    // angle to rotate image by (in radians)

#define MIN_EPSILON_ERROR 5e-3f

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

extern "C"
void computeGold(float *reference, float *idata, const unsigned int len);

static CUresult initCUDA(int argc, char **argv, CUfunction *);

const char *sSDKsample = "simpleTextureDrv (Driver API)";

//define input ptx file for different platforms
#if defined(_WIN64) || defined(__LP64__)
#define PTX_FILE "simpleTexture_kernel64.ptx"
#define CUBIN_FILE "simpleTexture_kernel64.cubin"
#else
#define PTX_FILE "simpleTexture_kernel32.ptx"
#define CUBIN_FILE "simpleTexture_kernel32.cubin"
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

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

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

////////////////////////////////////////////////////////////////////////////////
// Globals
////////////////////////////////////////////////////////////////////////////////
CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;

void
showHelp()
{
    printf("\n> [%s] Command line options\n", sSDKsample);
    printf("\t-device=n          (where n=0,1,2.... for the GPU device)\n\n");
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    if (checkCmdLineFlag(argc, (const char **)argv, "help"))
    {
        showHelp();
        return 0;
    }

    runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char **argv)
{
    bool bTestResults = true;

    // initialize CUDA
    CUfunction transform = NULL;

    if (initCUDA(argc, argv, &transform) != CUDA_SUCCESS)
    {
        exit(EXIT_FAILURE);
    }

    // load image from disk
    float *h_data = NULL;
    unsigned int width, height;
    char *image_path = sdkFindFilePath(image_filename, argv[0]);

    if (image_path == NULL)
    {
        printf("Unable to find image file: '%s'\n", image_filename);
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(image_path, &h_data, &width, &height);

    size_t       size = width * height * sizeof(float);
    printf("Loaded '%s', %d x %d pixels\n", image_filename, width, height);

    // load reference image from image (output)
    float *h_data_ref = (float *) malloc(size);
    char *ref_path = sdkFindFilePath(ref_filename, argv[0]);

    if (ref_path == NULL)
    {
        printf("Unable to find reference file %s\n", ref_filename);
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(ref_path, &h_data_ref, &width, &height);

    // allocate device memory for result
    CUdeviceptr d_data = (CUdeviceptr)NULL;
    checkCudaErrors(cuMemAlloc(&d_data, size));

    // allocate array and copy image data
    CUarray cu_array;
    CUDA_ARRAY_DESCRIPTOR desc;
    desc.Format = CU_AD_FORMAT_FLOAT;
    desc.NumChannels = 1;
    desc.Width = width;
    desc.Height = height;
    checkCudaErrors(cuArrayCreate(&cu_array, &desc));
    CUDA_MEMCPY2D copyParam;
    memset(&copyParam, 0, sizeof(copyParam));
    copyParam.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copyParam.dstArray = cu_array;
    copyParam.srcMemoryType = CU_MEMORYTYPE_HOST;
    copyParam.srcHost = h_data;
    copyParam.srcPitch = width * sizeof(float);
    copyParam.WidthInBytes = copyParam.srcPitch;
    copyParam.Height = height;
    checkCudaErrors(cuMemcpy2D(&copyParam));

    // set texture parameters
    CUtexref cu_texref;
    checkCudaErrors(cuModuleGetTexRef(&cu_texref, cuModule, "tex"));
    checkCudaErrors(cuTexRefSetArray(cu_texref, cu_array, CU_TRSA_OVERRIDE_FORMAT));
    checkCudaErrors(cuTexRefSetAddressMode(cu_texref, 0, CU_TR_ADDRESS_MODE_WRAP));
    checkCudaErrors(cuTexRefSetAddressMode(cu_texref, 1, CU_TR_ADDRESS_MODE_WRAP));
    checkCudaErrors(cuTexRefSetFilterMode(cu_texref, CU_TR_FILTER_MODE_LINEAR));
    checkCudaErrors(cuTexRefSetFlags(cu_texref, CU_TRSF_NORMALIZED_COORDINATES));
    checkCudaErrors(cuTexRefSetFormat(cu_texref, CU_AD_FORMAT_FLOAT, 1));

    checkCudaErrors(cuParamSetTexRef(transform, CU_PARAM_TR_DEFAULT, cu_texref));

    // There are two ways to launch CUDA kernels via the Driver API.
    // In this CUDA Sample, we illustrate both ways to pass parameters
    // and specify parameters.  By default we use the simpler method.
    int block_size = 8;
    StopWatchInterface *timer = NULL;

    if (1)
    {
        // This is the new CUDA 4.0 API for Kernel Parameter passing and Kernel Launching (simpler method)
        void *args[5] = { &d_data, &width, &height, &angle };

        checkCudaErrors(cuLaunchKernel(transform, (width/block_size), (height/block_size), 1,
                                       block_size     , block_size     , 1,
                                       0,
                                       NULL, args, NULL));
        checkCudaErrors(cuCtxSynchronize());
        sdkCreateTimer(&timer);
        sdkStartTimer(&timer);

        // launch kernel again for performance measurement
        checkCudaErrors(cuLaunchKernel(transform, (width/block_size), (height/block_size), 1,
                                       block_size     , block_size     , 1,
                                       0,
                                       NULL, args, NULL));
    }
    else
    {
        // This is the new CUDA 4.0 API for Kernel Parameter passing and Kernel Launching (advanced method)
        int offset = 0;
        char argBuffer[256];

        // pass in launch parameters (not actually de-referencing CUdeviceptr).  CUdeviceptr is
        // storing the value of the parameters
        *((CUdeviceptr *)&argBuffer[offset]) = d_data;
        offset += sizeof(d_data);
        *((unsigned int *)&argBuffer[offset]) = width;
        offset += sizeof(width);
        *((unsigned int *)&argBuffer[offset]) = height;
        offset += sizeof(height);
        *((float *)&argBuffer[offset]) = angle;
        offset += sizeof(angle);

        void *kernel_launch_config[5] =
        {
            CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
            CU_LAUNCH_PARAM_BUFFER_SIZE,    &offset,
            CU_LAUNCH_PARAM_END
        };

        // new CUDA 4.0 Driver API Kernel launch call (warmup)
        checkCudaErrors(cuLaunchKernel(transform, (width/block_size), (height/block_size), 1,
                                       block_size     , block_size     , 1,
                                       0,
                                       NULL, NULL, (void **)&kernel_launch_config));
        checkCudaErrors(cuCtxSynchronize());
        sdkCreateTimer(&timer);
        sdkStartTimer(&timer);

        // launch kernel again for performance measurement
        checkCudaErrors(cuLaunchKernel(transform, (width/block_size), (height/block_size), 1,
                                       block_size     , block_size     , 1,
                                       0, 0,
                                       NULL, (void **)&kernel_launch_config));
    }


    checkCudaErrors(cuCtxSynchronize());
    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    printf("%.2f Mpixels/sec\n", (width*height / (sdkGetTimerValue(&timer) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&timer);

    // allocate mem for the result on host side
    float *h_odata = (float *) malloc(size);
    // copy result from device to host
    checkCudaErrors(cuMemcpyDtoH(h_odata, d_data, size));

    // write result to file
    char output_filename[1024];
    strcpy(output_filename, image_path);
    strcpy(output_filename + strlen(image_path) - 4, "_out.pgm");
    sdkSavePGM(output_filename, h_odata, width, height);
    printf("Wrote '%s'\n", output_filename);

    // write regression file if necessary
    if (checkCmdLineFlag(argc, (const char **) argv, "regression"))
    {
        // write file for regression test
        sdkWriteFile<float>("./data/regression.dat", h_odata, width*height, 0.0f, false);
    }
    else
    {
        // We need to reload the data from disk, because it is inverted upon output
        sdkLoadPGM(output_filename, &h_odata, &width, &height);

        printf("Comparing files\n");
        printf("\toutput:    <%s>\n", output_filename);
        printf("\treference: <%s>\n", ref_path);
        bTestResults = compareData(h_odata, h_data_ref, width*height, MIN_EPSILON_ERROR, 0.15f);
    }

    // cleanup memory
    checkCudaErrors(cuMemFree(d_data));
    checkCudaErrors(cuArrayDestroy(cu_array));

    free(image_path);
    free(ref_path);

    checkCudaErrors(cuCtxDestroy(cuContext));

    exit(bTestResults ? EXIT_SUCCESS : EXIT_FAILURE);
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


////////////////////////////////////////////////////////////////////////////////
//! This initializes CUDA, and loads the *.ptx CUDA module containing the
//! kernel function.  After the module is loaded, cuModuleGetFunction
//! retrieves the CUDA function pointer "cuFunction"
////////////////////////////////////////////////////////////////////////////////
static CUresult
initCUDA(int argc, char **argv, CUfunction *transform)
{
    CUfunction cuFunction = 0;
    CUresult status;
    int major = 0, minor = 0, devID = 0;
    char deviceName[100];
    string module_path, ptx_source;

    cuDevice = findCudaDevice(argc, argv, &devID);

    // get compute capabilities and the devicename
    checkCudaErrors(cuDeviceComputeCapability(&major, &minor, cuDevice));
    checkCudaErrors(cuDeviceGetName(deviceName, 256, cuDevice));
    printf("> GPU Device has SM %d.%d compute capability\n", major, minor);

    status = cuCtxCreate(&cuContext, 0, cuDevice);

    if (CUDA_SUCCESS != status)
    {
        printf("cuCtxCreate(0) returned %d\n-> %s\n", status, getCudaDrvErrorString(status));
        goto Error;
    }

    // first search for the module_path before we try to load the results
    if (!findModulePath(PTX_FILE, module_path, argv, ptx_source))
    {
        if (!findModulePath(CUBIN_FILE, module_path, argv, ptx_source))
        {
            printf("> findModulePath could not find <simpleTexture_kernel> ptx or cubin\n");
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

    status = cuModuleGetFunction(&cuFunction, cuModule, "transformKernel");

    if (CUDA_SUCCESS != status)
    {
        goto Error;
    }

    *transform = cuFunction;

    return CUDA_SUCCESS;
Error:
    cuCtxDestroy(cuContext);
    return status;
}
