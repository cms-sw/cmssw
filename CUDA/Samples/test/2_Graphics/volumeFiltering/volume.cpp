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

// CUDA Runtime
#include <cuda_runtime.h>

// Helper functions
#include <helper_cuda.h>
#include <helper_math.h>
#include "volume.h"

void Volume_init(Volume *vol, cudaExtent dataSize, void *h_data, int allowStore)
{
    // create 3D array
    vol->channelDesc = cudaCreateChannelDesc<VolumeType>();
    checkCudaErrors(cudaMalloc3DArray(&vol->content, &vol->channelDesc, dataSize, allowStore ? cudaArraySurfaceLoadStore : 0));
    vol->size = dataSize;

    if (h_data)
    {
        // copy data to 3D array
        cudaMemcpy3DParms copyParams = {0};
        copyParams.srcPtr   = make_cudaPitchedPtr(h_data, dataSize.width*sizeof(VolumeType), dataSize.width, dataSize.height);
        copyParams.dstArray = vol->content;
        copyParams.extent   = dataSize;
        copyParams.kind     = cudaMemcpyHostToDevice;
        checkCudaErrors(cudaMemcpy3D(&copyParams));
    }
}

void Volume_deinit(Volume *vol)
{
    checkCudaErrors(cudaFreeArray(vol->content));
    vol->content = 0;
}

