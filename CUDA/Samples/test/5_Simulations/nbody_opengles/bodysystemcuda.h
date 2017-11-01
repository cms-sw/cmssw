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

#ifndef __BODYSYSTEMCUDA_H__
#define __BODYSYSTEMCUDA_H__

#include "bodysystem.h"

template <typename T>
struct DeviceData
{
    T *dPos[2]; // mapped host pointers
    T *dVel;
    cudaEvent_t  event;
    unsigned int offset;
    unsigned int numBodies;
};

// CUDA BodySystem: runs on the GPU
template <typename T>
class BodySystemCUDA : public BodySystem<T>
{
    public:
        BodySystemCUDA(unsigned int numBodies,
                       unsigned int numDevices,
                       unsigned int blockSize,
                       bool usePBO,
                       bool useSysMem = false);
        virtual ~BodySystemCUDA();

        virtual void loadTipsyFile(const std::string &filename);

        virtual void update(T deltaTime);

        virtual void setSoftening(T softening);
        virtual void setDamping(T damping);

        virtual T *getArray(BodyArray array);
        virtual void   setArray(BodyArray array, const T *data);

        virtual unsigned int getCurrentReadBuffer() const
        {
            return m_pbo[m_currentRead];
        }

        virtual unsigned int getNumBodies() const
        {
            return m_numBodies;
        }

    protected: // methods
        BodySystemCUDA() {}

        virtual void _initialize(int numBodies);
        virtual void _finalize();

    protected: // data
        unsigned int m_numBodies;
        unsigned int m_numDevices;
        bool m_bInitialized;

        // Host data
        T *m_hPos[2];
        T *m_hVel;

        DeviceData<T> *m_deviceData;

        bool m_bUsePBO;
        bool m_bUseSysMem;
        unsigned int m_SMVersion;

        T m_damping;

        unsigned int m_pbo[2];
        cudaGraphicsResource *m_pGRes[2];
        unsigned int m_currentRead;
        unsigned int m_currentWrite;

        unsigned int m_blockSize;
};

#include "bodysystemcuda_impl.h"

#endif // __BODYSYSTEMCUDA_H__
