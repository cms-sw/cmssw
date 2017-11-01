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

#ifndef __PARTICLESYSTEM_H__
#define __PARTICLESYSTEM_H__

#include <helper_functions.h>
#include "particles_kernel.cuh"
#include "vector_functions.h"
#include "GpuArray.h"
#include "nvMath.h"

using namespace nv;

// CUDA BodySystem: runs on the GPU
class ParticleSystem
{
    public:
        ParticleSystem(uint numParticles, bool bUseVBO = true, bool bUseGL = true);
        ~ParticleSystem();

        enum ParticleConfig
        {
            CONFIG_RANDOM,
            CONFIG_GRID,
            _NUM_CONFIGS
        };

        void step(float deltaTime);
        void depthSort();
        void reset(ParticleConfig config);

        uint getNumParticles()
        {
            return m_numParticles;
        }

        uint getPosBuffer()
        {
            return m_pos.getVbo();
        }
        uint getVelBuffer()
        {
            return m_vel.getVbo();
        }
        uint getColorBuffer()
        {
            return 0;
        }
        uint getSortedIndexBuffer()
        {
            return m_indices.getVbo();
        }
        uint *getSortedIndices();

        float getParticleRadius()
        {
            return m_particleRadius;
        }

        SimParams &getParams()
        {
            return m_params;
        }

        void setSorting(bool x)
        {
            m_doDepthSort = x;
        }
        void setModelView(float *m);
        void setSortVector(float3 v)
        {
            m_sortVector = v;
        }

        void addSphere(uint &index, vec3f pos, vec3f vel, int r, float spacing, float jitter, float lifetime);
        void discEmitter(uint &index, vec3f pos, vec3f vel, vec3f vx, vec3f vy, float r, int n, float lifetime, float lifetimeVariance);
        void sphereEmitter(uint &index, vec3f pos, vec3f vel, vec3f spread, float r, int n, float lifetime, float lifetimeVariance);

        void dumpParticles(uint start, uint count);
        void dumpBin(float4 **posData, float4 **velData);

    protected: // methods
        ParticleSystem() {}

        void _initialize(int numParticlesm, bool bUseGL=true);
        void _free();

        void initGrid(vec3f start, uint3 size, vec3f spacing, float jitter, vec3f vel, uint numParticles, float lifetime=100.0f);
        void initCubeRandom(vec3f origin, vec3f size, vec3f vel, float lifetime=100.0f);

    protected: // data
        bool m_bInitialized;
        bool m_bUseVBO;
        uint m_numParticles;

        float m_particleRadius;

        GpuArray<float4> m_pos;
        GpuArray<float4> m_vel;

        // params
        SimParams m_params;

        float4x4 m_modelView;
        float3 m_sortVector;
        bool m_doDepthSort;

        GpuArray<float> m_sortKeys;
        GpuArray<uint> m_indices;   // sorted indices for rendering

        StopWatchInterface *m_timer;
        float m_time;
};

#endif // __PARTICLESYSTEM_H__
