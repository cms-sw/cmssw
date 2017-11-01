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
* CUDA Device code for particle simulation.
*/

#ifndef _PARTICLES_KERNEL_H_
#define _PARTICLES_KERNEL_H_

#include "helper_math.h"
#include "math_constants.h"
#include "particles_kernel.cuh"

texture<float4, 3, cudaReadModeElementType> noiseTex;

// simulation parameters
__constant__ SimParams params;

// look up in 3D noise texture
__device__
float3 noise3D(float3 p)
{
    float4 n = tex3D(noiseTex, p.x, p.y, p.z);
    return make_float3(n.x, n.y, n.z);
}

// integrate particle attributes
struct integrate_functor
{
    float deltaTime;

    __host__ __device__
    integrate_functor(float delta_time) : deltaTime(delta_time) {}

    template <typename Tuple>
    __device__
    void operator()(Tuple t)
    {
        volatile float4 posData = thrust::get<2>(t);
        volatile float4 velData = thrust::get<3>(t);

        float3 pos = make_float3(posData.x, posData.y, posData.z);
        float3 vel = make_float3(velData.x, velData.y, velData.z);

        // update particle age
        float age = posData.w;
        float lifetime = velData.w;

        if (age < lifetime)
        {
            age += deltaTime;
        }
        else
        {
            age = lifetime;
        }

        // apply accelerations
        vel += params.gravity * deltaTime;

        // apply procedural noise
        float3 noise = noise3D(pos*params.noiseFreq + params.time*params.noiseSpeed);
        vel += noise * params.noiseAmp;

        // new position = old position + velocity * deltaTime
        pos += vel * deltaTime;

        vel *= params.globalDamping;

        // store new position and velocity
        thrust::get<0>(t) = make_float4(pos, age);
        thrust::get<1>(t) = make_float4(vel, velData.w);
    }
};

struct calcDepth_functor
{
    float3 sortVector;

    __host__ __device__
    calcDepth_functor(float3 sort_vector) : sortVector(sort_vector) {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        volatile float4 p = thrust::get<0>(t);
        float key = -dot(make_float3(p.x, p.y, p.z), sortVector); // project onto sort vector
        thrust::get<1>(t) = key;
    }
};

#endif
