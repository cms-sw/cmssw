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

/* cudaModuleMgr has the C and C++ implementations.  These functions manage:
 *   loading CUBIN/PTX
 *    initializing CUBIN/PTX kernel function,
 *    mapping CUDA kernel function pointers,
 *    and obtaining global memory pointers and texture references
 */

#ifndef _CUDAMODULEMGR_H_
#define _CUDAMODULEMGR_H_

#include <memory>
#include <iostream>
#include <cassert>

// Function String Name and a pointer to the CUDA Kernel Function
typedef struct CudaKernels
{
    CUfunction fpCuda;
    std::string func_name;
} *pCudaKernels_;

// Global Memory Name and the device pointer
typedef struct CudaGlobalMem
{
    CUdeviceptr  devicePtr;
    size_t       nBytes;
    std::string  address_name;
} *pGlobalMem_;

// TexReference and the CUtexref pointer
typedef struct CudaTexRef
{
    CUtexref     texRef;
    unsigned int nBytes;
    std::string  texref_name;
} *pTexRef_;

// We have a C++ and a C version so that developers can choose to use either depending on their preferences
typedef struct _sCUModuleContext
{
    std::string mModuleName;

    int nMaxKernels_;    // maximum number of kernels
    int nMaxGlobalMem_;  // maximum number of global constants
    int nMaxTexRef_;     // maximum number of texture references

    int nLastKernel_;    // the last kernel
    int nLastGlobalMem_; // the last global constant used
    int nLastTexRef_;    // the last texture reference used

    CudaKernels    *pCudaKernels_;  // stores the data, strings for the CUDA kernels
    CudaGlobalMem *pGlobalMem_;     // stores the data, strings for the Global Memory (Device Pointers)
    CudaTexRef     *pTexRef_;       // stores the data, strings for the Texture References

    CUmodule    cuModule_;
} sCtxModule;

// Here is the C implementation for the Module Manager, the C++ class calls the C implementation
extern "C" bool     modInitCtx(sCtxModule *mCtx, const char *filename, const char *exec_path, int nKernels, int nGlobalMem, int nTexRef);
extern "C" void     modFreeCtx(sCtxModule *mCtx);

extern "C" CUresult modGetCudaDevicePtr(sCtxModule *mCtx, const char *address_name, CUdeviceptr *pGlobalMem);
extern "C" CUresult modGetTexRef(sCtxModule *mCtx, const char *texref_name,  CUtexref    *pTexRef);
extern "C" CUresult modLaunchKernel(sCtxModule *mCtx, CUfunction fpFunc, dim3 block, dim3 grid);

extern "C" int      modFindIndex_CudaKernels(sCtxModule *mCtx, const char *func_name);
extern "C" int      modFindIndex_GlobalMem(sCtxModule *mCtx, const char *address_name);
extern "C" int      modFindIndex_TexRef(sCtxModule *mCtx, const char *texref_name);


// Here is the C++ Class interface to the Module Manager
class CUmoduleManager
{
    public:
        // For each CUBIN file loaded, one CUBIN is associated with one CUmodule
        CUmoduleManager(const char *filename_module, const char *exec_path, int nKernels, int nGlobalMem, int nTexRef);
        ~CUmoduleManager();

        CUresult GetCudaFunction(const char *func_name,    CUfunction  *fpCudaKernel = 0);
        CUresult GetCudaDevicePtr(const char *address_name, CUdeviceptr *pGlobalMem = 0);
        CUresult GetTexRef(const char *texref_name,  CUtexref    *pTexRef = 0);

        int findIndex_CudaKernels(const char *func_name);
        int findIndex_GlobalMem(const char *address_name);
        int findIndex_TexRef(const char *texref_name);

        CUresult launchKernel(CUfunction fpFunc, dim3 block, dim3 grid);

        CUmodule getModule()
        {
            return mCtx.cuModule_;
        }

    protected:
        // This stores all of the relevant data for the Module (PTX or CUBIN)
        sCtxModule mCtx;
};


#endif
