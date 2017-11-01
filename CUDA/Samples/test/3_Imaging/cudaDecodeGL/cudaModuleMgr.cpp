/*
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* CUmoduleManager manages loading CUBIN, initializing CUBIN kernel function,
 * initializing CUDA kernel function pointers, and obtaining global memory
 * addresses (i.e. constants).
 */

#include <stdio.h>
#include <string.h>

#include <iostream>
#include <cstring>

#include "dynlink_cuda.h"
#include "dynlink_builtin_types.h"
#include "helper_cuda_drvapi.h"
#include "cudaModuleMgr.h"

#define ERROR_BUFFER_SIZE 256

using namespace std;

// CUDA Module Manager (C implementation)
//      filename_module - CUDA or PTX file path
//      exec_path       - execution path
//      nKernels        - total # of different CUDA kernel functions in the CUBIN/OTX
//      nGlobalMem      - total # of Global Memory arrays defined in the CUBIN/PTX (i.e. constants)
//      nTexRef         - total # of Texture References arrays defined in the CUBIN/PTX (i.e. texture arrays)
extern "C"
bool modInitCTX(sCtxModule *pCtx, const char *filename, const char *exec_path, int nKernels, int nGlobalMem, int nTexRef)
{
    pCtx->nMaxKernels_   = nKernels;
    pCtx->nMaxGlobalMem_ = nGlobalMem;
    pCtx->nMaxTexRef_    = nTexRef;
    pCtx->mModuleName    = filename;

    CUresult cuStatus;
    string module_path;
    string ptx_source;

    printf("\nstring = %s\n", pCtx->mModuleName.c_str());
    char *actual_path = sdkFindFilePath(pCtx->mModuleName.c_str(), exec_path);

    if (actual_path)
    {
        module_path = actual_path;
    }
    else
    {
        printf(">> modInitCTX() <%36s> not found!\n", pCtx->mModuleName.c_str());
        return false;
    }

    if (module_path.empty())
    {
        printf(">> modInitCTX() <%36s> not found!\n", pCtx->mModuleName.c_str());
        return false;
    }
    else
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
        delete [] buf;
    }

    if (pCtx->mModuleName.rfind(".ptx") != string::npos)
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

        cuStatus = cuModuleLoadDataEx(&pCtx->cuModule_, ptx_source.c_str(), jitNumOptions, jitOptions, (void **)jitOptVals);

        if (cuStatus != CUDA_SUCCESS)
        {
            printf("cuModuleLoadDataEx error!\n");
        }
//      printf("> PTX JIT log:\n%s\n", jitLogBuffer);
        delete[] jitOptions;
        delete[] jitOptVals;
        delete[] jitLogBuffer;
    }
    else
    {
        cuStatus = cuModuleLoad(&pCtx->cuModule_, module_path.c_str());

        if (cuStatus != CUDA_SUCCESS)
        {
            printf("cuModuleLoad error!\n");
        }
    }

    printf(">> modInitCTX<%20s > initialized %s",
           pCtx->mModuleName.c_str(), (cuStatus == CUDA_SUCCESS) ? "OK\n" : "FAILED\n");

    if (cuStatus == CUDA_SUCCESS)
    {
        pCtx->pCudaKernels_  = new CudaKernels  [pCtx->nMaxKernels_  ];
        pCtx->nLastKernel_    = 0;
        pCtx->pGlobalMem_    = new CudaGlobalMem[pCtx->nMaxGlobalMem_];
        pCtx->nLastGlobalMem_ = 0;
        pCtx->pTexRef_       = new CudaTexRef   [pCtx->nMaxTexRef_   ];
        pCtx->nLastTexRef_    = 0;
    }

    return (cuStatus == CUDA_SUCCESS) ? true : false;
}

extern "C"
void modFreeCTX(sCtxModule *pCtx)
{
    // release memory allocated
    delete [] pCtx->pCudaKernels_;
    delete [] pCtx->pGlobalMem_;
    delete [] pCtx->pTexRef_;

    // release the cuModule and resource strings
    cuModuleUnload(pCtx->cuModule_);

    printf(">> modFreeCTX success!\n");
}

extern "C"
CUresult modGetCudaFunction(sCtxModule *pCtx, const char *func_name,    CUfunction *fpCudaKernel)
{
    int idx = pCtx->nLastKernel_;

    CUresult cuStatus = cuModuleGetFunction(&(pCtx->pCudaKernels_[idx].fpCuda), pCtx->cuModule_, func_name);
    printf(">> modGetCudaFunction< CUDA file: %36s >\n", pCtx->mModuleName.c_str());
#if defined(_MSC_VER_)
    printf("   CUDA Kernel Function (0x%08Ix) = <%20s >\n", (size_t)(pCtx->pCudaKernels_[idx].fpCuda), func_name);
#else
    printf("   CUDA Kernel Function (0x%08zx) = <%20s >\n", (size_t)(pCtx->pCudaKernels_[idx].fpCuda), func_name);
#endif
    pCtx->pCudaKernels_[idx].func_name = func_name;

    if (fpCudaKernel)
        *fpCudaKernel = pCtx->pCudaKernels_[idx].fpCuda;

    pCtx->nLastKernel_++;

    return cuStatus;
}

extern "C"
CUresult modGetCudaDevicePtr(sCtxModule *pCtx, const char *address_name, CUdeviceptr *pGlobalMem)
{
    int idx = pCtx->nLastGlobalMem_;

    CUresult cuStatus = cuModuleGetGlobal(&(pCtx->pGlobalMem_[idx].devicePtr),  &(pCtx->pGlobalMem_[idx].nBytes), pCtx->cuModule_, address_name);
    printf(">> modGetCudaDevicePtr<%36s >\n", pCtx->mModuleName.c_str());
    printf("   CUDA Device Memory (0x%08x) <%24s >\n", (unsigned int)(pCtx->pGlobalMem_[idx].devicePtr), address_name);
    pCtx->pGlobalMem_[idx].address_name = address_name;

    if (pGlobalMem)
        *pGlobalMem = pCtx->pGlobalMem_[idx].devicePtr;

    pCtx->nLastGlobalMem_++;

    return cuStatus;
}

extern "C"
CUresult modGetTexRef(sCtxModule *pCtx, const char *texref_name,  CUtexref    *pTexRef)
{
    int idx = pCtx->nLastTexRef_;

    CUresult cuStatus = cuModuleGetTexRef(&(pCtx->pTexRef_[idx].texRef), pCtx->cuModule_, texref_name);
    printf(">> modGetTexRef<%36s>\n", pCtx->mModuleName.c_str());
#if defined(_MSC_VER_)
    printf("   CUDA TextureReference (0x%08Ix) <24%s >\n", (size_t)(pCtx->pTexRef_[idx].texRef), texref_name);
#else
    printf("   CUDA TextureReference (0x%08zx) <24%s >\n", (size_t)(pCtx->pTexRef_[idx].texRef), texref_name);
#endif
    pCtx->pTexRef_[idx].texref_name = texref_name;

    if (pTexRef)
        *pTexRef = pCtx->pTexRef_[idx].texRef;

    pCtx->nLastTexRef_++;

    return cuStatus;

}

extern "C"
int      modFindIndex_CudaKernels(sCtxModule *pCtx, const char *func_name)
{
    int found = -1;

    for (int i=0; i < pCtx->nLastKernel_; i++)
    {
        if (pCtx->pCudaKernels_[i].func_name.compare(func_name))
        {
            found = i;
            break;
        }
    }

    return found;

}

extern "C"
int      modFindIndex_GlobalMem(sCtxModule *pCtx, const char *address_name)
{
    int found = -1;

    for (int i=0; i < pCtx->nLastGlobalMem_; i++)
    {
        if (pCtx->pGlobalMem_[i].address_name.compare(address_name))
        {
            found = i;
            break;
        }
    }

    return found;
}

extern "C"
int      modFindIndex_TexRef(sCtxModule *pCtx, const char *texref_name)
{
    int found = -1;

    for (int i=0; i < pCtx->nLastGlobalMem_; i++)
    {
        if (pCtx->pTexRef_[i].texref_name.compare(texref_name))
        {
            found = i;
            break;
        }
    }

    return found;
}

extern "C"
CUresult modLaunchKernel(sCtxModule *pCtx, CUfunction fpFunc, dim3 block, dim3 grid)
{
    CUresult error = CUDA_SUCCESS;
    /*
        dim3 block(32,16);
        dim3 grid((width+(2*block.x-1))/(2*block.x), (height+(block.y-1))/block.y);

        // setup execution parameters
        cutilDrvSafeCall(cuFuncSetBlockShape( fpFunc, block.x, block.y, 1 ));
        int offset = 0;
        cutilDrvSafeCall(cuParamSeti        ( fpFunc, 0,  d_srcNV12 ));     offset += sizeof(d_srcNV12);
        cutilDrvSafeCall(cuParamSeti        ( fpFunc, 4,  nSourcePitch ));  offset += sizeof(nSourcePitch);
        cutilDrvSafeCall(cuParamSeti        ( fpFunc, 8,  d_dstARGB ));     offset += sizeof(d_dstARGB);
        cutilDrvSafeCall(cuParamSeti        ( fpFunc, 12, nDestPitch ));    offset += sizeof(nDestPitch);
        cutilDrvSafeCall(cuParamSeti        ( fpFunc, 16, width ));         offset += sizeof(width);
        cutilDrvSafeCall(cuParamSeti        ( fpFunc, 20, height ));        offset += sizeof(height);
        cutilDrvSafeCall(cuParamSetSize     ( fpFunc,     offset));

        error = cuLaunchGrid( fpFunc, grid.x, grid.y );
    */
    return error;
}


// CUDA Module Manager (C++ implementation)
//      filename_module - CUDA or PTX file path
//      exec_path       - execution path
//      nKernels        - total # of different CUDA kernel functions in the CUBIN/OTX
//      nGlobalMem      - total # of Global Memory arrays defined in the CUBIN/PTX (i.e. constants)
//      nTexRef         - total # of Texture References arrays defined in the CUBIN/PTX (i.e. texture arrays)
CUmoduleManager::CUmoduleManager(const char *filename_module,
                                 const char *exec_path,
                                 int nKernels,
                                 int nGlobalMem,
                                 int nTexRef)
{
    if (modInitCTX(&mCtx, filename_module, exec_path, nKernels, nGlobalMem, nTexRef) == false)
    {
        throw (filename_module);
    }
}

CUmoduleManager::~CUmoduleManager()
{
    modFreeCTX(&mCtx);
}

// This gets the CUDA kernel function pointers from the CUBIN/PTX
CUresult
CUmoduleManager::GetCudaFunction(const char *func_name, CUfunction *fpCudaKernel)
{
    return modGetCudaFunction(&mCtx, func_name, fpCudaKernel);
}

// This gets the CUDA device pointers from the CUBIN/PTX
CUresult
CUmoduleManager::GetCudaDevicePtr(const char *address_name, CUdeviceptr *pGlobalMem)
{
    return modGetCudaDevicePtr(&mCtx, address_name, pGlobalMem);
}

// This retrieves the CUDA Texture References from the CUBIN/PTX
CUresult
CUmoduleManager::GetTexRef(const char *texref_name, CUtexref *pTexRef)
{
    return modGetTexRef(&mCtx, texref_name, pTexRef);
}

// This retrieves the CUDA Kernel Function from the CUBIN/PTX
int CUmoduleManager::findIndex_CudaKernels(const char *func_name)
{
    return modFindIndex_CudaKernels(&mCtx, func_name);
}

// This retrieves the CUDA Global Memory functions from the CUBIN/PTX
int CUmoduleManager::findIndex_GlobalMem(const char *address_name)
{
    return modFindIndex_GlobalMem(&mCtx, address_name);
}

int CUmoduleManager::findIndex_TexRef(const char *texref_name)
{
    return modFindIndex_TexRef(&mCtx, texref_name);

}

// TODO figure out how to do the same thing the CUDA Runtime did in a C++ style
CUresult CUmoduleManager::launchKernel(CUfunction fpFunc, dim3 block, dim3 grid)
{
    return modLaunchKernel(&mCtx, fpFunc, block, grid);
}
