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
This file contains simple wrapper functions that call the CUDA kernels
*/
#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include <helper_gl.h>
#include <helper_cuda.h>
#include <cstdlib>
#include <cstdio>
#include <string.h>
#include <cuda_gl_interop.h>

#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

#include "particles_kernel_device.cuh"
#include "ParticleSystem.cuh"

extern "C"
{

    cudaArray *noiseArray;

    void initCuda(bool bUseGL)
    {
        if (bUseGL)
        {
            cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
        }
        else
        {
            cudaSetDevice(gpuGetMaxGflopsDeviceId());
        }
    }

    void setParameters(SimParams *hostParams)
    {
        // copy parameters to constant memory
        checkCudaErrors(cudaMemcpyToSymbol(params, hostParams, sizeof(SimParams)));
    }

    //Round a / b to nearest higher integer value
    int iDivUp(int a, int b)
    {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }

    // compute grid and thread block size for a given number of elements
    void computeGridSize(int n, int blockSize, int &numBlocks, int &numThreads)
    {
        numThreads = min(blockSize, n);
        numBlocks = iDivUp(n, numThreads);
    }

    inline float frand()
    {
        return rand() / (float) RAND_MAX;
    }

    // create 3D texture containing random values
    void createNoiseTexture(int w, int h, int d)
    {
        cudaExtent size = make_cudaExtent(w, h, d);
        size_t elements = size.width*size.height*size.depth;

        float *volumeData = (float *)malloc(elements*4*sizeof(float));
        float *ptr = volumeData;

        for (size_t i=0; i<elements; i++)
        {
            *ptr++ = frand()*2.0f-1.0f;
            *ptr++ = frand()*2.0f-1.0f;
            *ptr++ = frand()*2.0f-1.0f;
            *ptr++ = frand()*2.0f-1.0f;
        }


        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
        checkCudaErrors(cudaMalloc3DArray(&noiseArray, &channelDesc, size));

        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr   = make_cudaPitchedPtr((void *)volumeData, size.width*sizeof(float4), size.width, size.height);
        copyParams.dstArray = noiseArray;
        copyParams.extent   = size;
        copyParams.kind     = cudaMemcpyHostToDevice;
        checkCudaErrors(cudaMemcpy3D(&copyParams));

        free(volumeData);

        // set texture parameters
        noiseTex.normalized = true;                      // access with normalized texture coordinates
        noiseTex.filterMode = cudaFilterModeLinear;      // linear interpolation
        noiseTex.addressMode[0] = cudaAddressModeWrap;   // wrap texture coordinates
        noiseTex.addressMode[1] = cudaAddressModeWrap;
        noiseTex.addressMode[2] = cudaAddressModeWrap;

        // bind array to 3D texture
        checkCudaErrors(cudaBindTextureToArray(noiseTex, noiseArray, channelDesc));
    }

    void
    integrateSystem(float4 *oldPos, float4 *newPos,
                    float4 *oldVel, float4 *newVel,
                    float deltaTime,
                    int numParticles)
    {
        thrust::device_ptr<float4> d_newPos(newPos);
        thrust::device_ptr<float4> d_newVel(newVel);
        thrust::device_ptr<float4> d_oldPos(oldPos);
        thrust::device_ptr<float4> d_oldVel(oldVel);

        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(d_newPos, d_newVel, d_oldPos, d_oldVel)),
            thrust::make_zip_iterator(thrust::make_tuple(d_newPos+numParticles, d_newVel+numParticles, d_oldPos+numParticles, d_oldVel+numParticles)),
            integrate_functor(deltaTime));
    }

    void
    calcDepth(float4  *pos,
              float   *keys,        // output
              uint    *indices,     // output
              float3   sortVector,
              int      numParticles)
    {
        thrust::device_ptr<float4> d_pos(pos);
        thrust::device_ptr<float> d_keys(keys);
        thrust::device_ptr<uint> d_indices(indices);

        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(d_pos, d_keys)),
            thrust::make_zip_iterator(thrust::make_tuple(d_pos+numParticles, d_keys+numParticles)),
            calcDepth_functor(sortVector));

        thrust::sequence(d_indices, d_indices + numParticles);
    }

    void sortParticles(float *sortKeys, uint *indices, uint numParticles)
    {
        thrust::sort_by_key(thrust::device_ptr<float>(sortKeys),
                            thrust::device_ptr<float>(sortKeys + numParticles),
                            thrust::device_ptr<uint>(indices));
    }

}   // extern "C"
