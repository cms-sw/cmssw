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

#include "particles_kernel.cuh"

extern "C"
{
    void initCuda(bool bUseGL);
    void setParameters(SimParams *hostParams);
    void createNoiseTexture(int w, int h, int d);

    void
    integrateSystem(float4 *oldPos, float4 *newPos,
                    float4 *oldVel, float4 *newVel,
                    float deltaTime,
                    int numParticles);

    void
    calcDepth(float4  *pos,
              float   *keys,        // output
              uint    *indices,     // output
              float3   sortVector,
              int      numParticles);

    void sortParticles(float *sortKeys, uint *indices, uint numParticles);
}

