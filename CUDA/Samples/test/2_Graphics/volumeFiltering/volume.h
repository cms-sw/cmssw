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

#ifndef _VOLUME_H_
#define _VOLUME_H_

#include <cuda_runtime.h>

typedef unsigned char VolumeType;

extern "C" {

    struct Volume
    {
        cudaArray            *content;
        cudaExtent            size;
        cudaChannelFormatDesc channelDesc;
    };

    void Volume_init(Volume *vol, cudaExtent size, void *data, int allowStore);
    void Volume_deinit(Volume *vol);

};

//////////////////////////////////////////////////////////////////////////

#ifdef __CUDACC__

/* Helper class to do popular integer storage to float conversions if required */

template< typename T >
struct VolumeTypeInfo
{};

template< >
struct VolumeTypeInfo<unsigned char>
{
    static const cudaTextureReadMode readMode = cudaReadModeNormalizedFloat;
    static __inline__ __device__ unsigned char convert(float sampled)
    {
        return (unsigned char)(saturate(sampled) * 255.0);
    }
};

template< >
struct VolumeTypeInfo<unsigned short>
{
    static const cudaTextureReadMode readMode = cudaReadModeNormalizedFloat;
    static __inline__ __device__ unsigned short convert(float sampled)
    {
        return (unsigned short)(saturate(sampled) * 65535.0);
    }
};

template< >
struct VolumeTypeInfo<float>
{
    static const cudaTextureReadMode readMode = cudaReadModeElementType;
    static __inline__ __device__ float convert(float sampled)
    {
        return sampled;
    }
};

#endif

#endif

