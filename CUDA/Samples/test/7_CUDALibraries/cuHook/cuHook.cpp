/*
* Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.  This source code is a "commercial item" as
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer software" and "commercial computer software
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*/

// This sample demonstrates how to use the CUDA hook library to receive callbacks

#include <cuda.h>
#include <stdio.h>
#include <dlfcn.h>

#include "libcuhook.h"

#define ASSERT_COND(x, msg)                                             \
    do {                                                                \
        if (!(x)) {                                                     \
            fprintf(stderr, "Error: Condition (%s) failed at %s:%d\n", #x, __FILE__, __LINE__); \
            fprintf(stderr, "cuHook sample failed (%s)\n", msg);        \
            exit(1);                                                    \
        }                                                               \
    } while (0)

/*
** Example of how to use the CUDA Interception Library, libcuhook.so
** The library has to be loaded via LD_PRELOAD, e.g. LD_PRELOAD=<full_path>/libcuhook.so.1 ./cuHook
*/

static int allocation_cb = 0;
static int free_cb = 0;
static int destroy_ctx_cb = 0;

CUresult device_allocation_callback(CUdeviceptr *dptr, size_t bytesize)
{
    fprintf(stdout, "Received memory allocation callback!\n");
    allocation_cb++;
    return CUDA_SUCCESS;
}

CUresult device_free_callback(CUdeviceptr dptr)
{
    fprintf(stdout, "Received memory de-allocation callback!\n");
    free_cb++;
    return CUDA_SUCCESS;
}

CUresult destroy_context_callback(CUcontext ctx)
{
    fprintf(stdout, "Received context destroy event!\n");
    destroy_ctx_cb++;
    return CUDA_SUCCESS;
}

int main()
{
    int count;
    CUcontext ctx;

    count = 0;

    cuInit(0);
    cuDeviceGetCount(&count);
    ASSERT_COND(count > 0, "No suitable devices found");

    // Load the cudaHookRegisterCallback symbol using the default library search order.
    // If we found the symbol, then the hooking library has been loaded
    fnCuHookRegisterCallback cuHook = (fnCuHookRegisterCallback)dlsym(RTLD_DEFAULT, "cuHookRegisterCallback");
    //    ASSERT_COND(cuHook, dlerror());
    if (cuHook) {
        // CUDA Runtime symbols cannot be hooked but the underlying driver ones _can_.
        // Example:
        // - cudaFree() will trigger cuMemFree
        // - cudaDeviceReset() will trigger a context change and you would need to intercept cuCtxGetCurrent/cuCtxSetCurrent
        cuHook(CU_HOOK_MEM_ALLOC, POST_CALL_HOOK, (void*)device_allocation_callback);
        cuHook(CU_HOOK_MEM_FREE, PRE_CALL_HOOK, (void*)device_free_callback);
        cuHook(CU_HOOK_CTX_DESTROY, POST_CALL_HOOK, (void*)destroy_context_callback);
    }

    cuCtxCreate(&ctx, 0, 0);
    {
        CUresult status;
        CUdeviceptr dptr;

        status = cuMemAlloc(&dptr, 1024);
        ASSERT_COND(status == CUDA_SUCCESS, "cuMemAlloc call failed");

        status = cuMemFree(dptr);
        ASSERT_COND(status == CUDA_SUCCESS, "cuMemFree call failed");
    }
    cuCtxDestroy(ctx);

    ASSERT_COND(allocation_cb == 1, "Didn't receive the allocation callback");
    ASSERT_COND(free_cb == 1, "Didn't receive the free callback");
    ASSERT_COND(destroy_ctx_cb == 1, "Didn't receive the destroy context callback");

    fprintf(stdout, "Sample finished successfully.\n");
    return (0);
}
