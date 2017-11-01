/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include "bodysystemcpu.h"

#include <assert.h>
#include <memory.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <helper_cuda.h>
#include <algorithm>
#include "tipsy.h"

#ifdef OPENMP
#include <omp.h>
#endif

template <typename T>
BodySystemCPU<T>::BodySystemCPU(int numBodies)
    : m_numBodies(numBodies),
      m_bInitialized(false),
      m_force(0),
      m_softeningSquared(.00125f),
      m_damping(0.995f)
{
    m_pos = 0;
    m_vel = 0;

    _initialize(numBodies);
}

template <typename T>
BodySystemCPU<T>::~BodySystemCPU()
{
    _finalize();
    m_numBodies = 0;
}

template <typename T>
void BodySystemCPU<T>::_initialize(int numBodies)
{
    assert(!m_bInitialized);

    m_numBodies = numBodies;

    m_pos    = new T[m_numBodies*4];
    m_vel    = new T[m_numBodies*4];
    m_force  = new T[m_numBodies*3];

    memset(m_pos,   0, m_numBodies*4*sizeof(T));
    memset(m_vel,   0, m_numBodies*4*sizeof(T));
    memset(m_force, 0, m_numBodies*3*sizeof(T));

    m_bInitialized = true;
}

template <typename T>
void BodySystemCPU<T>::_finalize()
{
    assert(m_bInitialized);

    delete [] m_pos;
    delete [] m_vel;
    delete [] m_force;

    m_bInitialized = false;
}

template <typename T>
void BodySystemCPU<T>::loadTipsyFile(const std::string &filename)
{
    if (m_bInitialized)
        _finalize();

    vector< typename vec4<T>::Type > positions;
    vector< typename vec4<T>::Type > velocities;
    vector< int> ids;

    int nBodies = 0;
    int nFirst=0, nSecond=0, nThird=0;

    read_tipsy_file(positions,
                    velocities,
                    ids,
                    filename,
                    nBodies,
                    nFirst,
                    nSecond,
                    nThird);

    _initialize(nBodies);

    memcpy(m_pos, &positions[0], sizeof(vec4<T>)*nBodies);
    memcpy(m_vel, &velocities[0], sizeof(vec4<T>)*nBodies);
}

template <typename T>
void BodySystemCPU<T>::update(T deltaTime)
{
    assert(m_bInitialized);

    _integrateNBodySystem(deltaTime);

    //std::swap(m_currentRead, m_currentWrite);
}

template <typename T>
T *BodySystemCPU<T>::getArray(BodyArray array)
{
    assert(m_bInitialized);

    T *data = 0;

    switch (array)
    {
        default:
        case BODYSYSTEM_POSITION:
            data = m_pos;
            break;

        case BODYSYSTEM_VELOCITY:
            data = m_vel;
            break;
    }

    return data;
}

template <typename T>
void BodySystemCPU<T>::setArray(BodyArray array, const T *data)
{
    assert(m_bInitialized);

    T *target = 0;

    switch (array)
    {
        default:
        case BODYSYSTEM_POSITION:
            target = m_pos;
            break;

        case BODYSYSTEM_VELOCITY:
            target = m_vel;
            break;
    }

    memcpy(target, data, m_numBodies*4*sizeof(T));
}

template<typename T>
T sqrt_T(T x)
{
    return sqrt(x);
}

template<>
float sqrt_T<float>(float x)
{
    return sqrtf(x);
}

template <typename T>
void bodyBodyInteraction(T accel[3], T posMass0[4], T posMass1[4], T softeningSquared)
{
    T r[3];

    // r_01  [3 FLOPS]
    r[0] = posMass1[0] - posMass0[0];
    r[1] = posMass1[1] - posMass0[1];
    r[2] = posMass1[2] - posMass0[2];

    // d^2 + e^2 [6 FLOPS]
    T distSqr = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    distSqr += softeningSquared;

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    T invDist = (T)1.0 / (T)sqrt((double)distSqr);
    T invDistCube =  invDist * invDist * invDist;

    // s = m_j * invDistCube [1 FLOP]
    T s = posMass1[3] * invDistCube;

    // (m_1 * r_01) / (d^2 + e^2)^(3/2)  [6 FLOPS]
    accel[0] += r[0] * s;
    accel[1] += r[1] * s;
    accel[2] += r[2] * s;
}

template <typename T>
void BodySystemCPU<T>::_computeNBodyGravitation()
{
#ifdef OPENMP
    #pragma omp parallel for
#endif

    for (int i = 0; i < m_numBodies; i++)
    {
        int indexForce = 3*i;

        T acc[3] = {0, 0, 0};

        // We unroll this loop 4X for a small performance boost.
        int j = 0;

        while (j < m_numBodies)
        {
            bodyBodyInteraction<T>(acc, &m_pos[4*i], &m_pos[4*j], m_softeningSquared);
            j++;
            bodyBodyInteraction<T>(acc, &m_pos[4*i], &m_pos[4*j], m_softeningSquared);
            j++;
            bodyBodyInteraction<T>(acc, &m_pos[4*i], &m_pos[4*j], m_softeningSquared);
            j++;
            bodyBodyInteraction<T>(acc, &m_pos[4*i], &m_pos[4*j], m_softeningSquared);
            j++;
        }

        m_force[indexForce  ] = acc[0];
        m_force[indexForce+1] = acc[1];
        m_force[indexForce+2] = acc[2];
    }
}

template <typename T>
void BodySystemCPU<T>::_integrateNBodySystem(T deltaTime)
{
    _computeNBodyGravitation();

#ifdef OPENMP
    #pragma omp parallel for
#endif

    for (int i = 0; i < m_numBodies; ++i)
    {
        int index = 4*i;
        int indexForce = 3*i;


        T pos[3], vel[3], force[3];
        pos[0] = m_pos[index+0];
        pos[1] = m_pos[index+1];
        pos[2] = m_pos[index+2];
        T invMass = m_pos[index+3];

        vel[0] = m_vel[index+0];
        vel[1] = m_vel[index+1];
        vel[2] = m_vel[index+2];

        force[0] = m_force[indexForce+0];
        force[1] = m_force[indexForce+1];
        force[2] = m_force[indexForce+2];

        // acceleration = force / mass;
        // new velocity = old velocity + acceleration * deltaTime
        vel[0] += (force[0] * invMass) * deltaTime;
        vel[1] += (force[1] * invMass) * deltaTime;
        vel[2] += (force[2] * invMass) * deltaTime;

        vel[0] *= m_damping;
        vel[1] *= m_damping;
        vel[2] *= m_damping;

        // new position = old position + velocity * deltaTime
        pos[0] += vel[0] * deltaTime;
        pos[1] += vel[1] * deltaTime;
        pos[2] += vel[2] * deltaTime;

        m_pos[index+0] = pos[0];
        m_pos[index+1] = pos[1];
        m_pos[index+2] = pos[2];

        m_vel[index+0] = vel[0];
        m_vel[index+1] = vel[1];
        m_vel[index+2] = vel[2];
    }
}
