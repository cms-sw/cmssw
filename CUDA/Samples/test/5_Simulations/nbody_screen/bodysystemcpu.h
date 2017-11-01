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

#ifndef __BODYSYSTEMCPU_H__
#define __BODYSYSTEMCPU_H__

#include "bodysystem.h"

// CPU Body System
template <typename T>
class BodySystemCPU : public BodySystem<T>
{
    public:
        BodySystemCPU(int numBodies);
        virtual ~BodySystemCPU();

        virtual void loadTipsyFile(const std::string &filename);

        virtual void update(T deltaTime);

        virtual void setSoftening(T softening)
        {
            m_softeningSquared = softening * softening;
        }
        virtual void setDamping(T damping)
        {
            m_damping = damping;
        }

        virtual T *getArray(BodyArray array);
        virtual void   setArray(BodyArray array, const T *data);

        virtual unsigned int getCurrentReadBuffer() const
        {
            return 0;
        }

        virtual unsigned int getNumBodies() const
        {
            return m_numBodies;
        }

    protected: // methods
        BodySystemCPU() {} // default constructor

        virtual void _initialize(int numBodies);
        virtual void _finalize();

        void _computeNBodyGravitation();
        void _integrateNBodySystem(T deltaTime);

    protected: // data
        int m_numBodies;
        bool         m_bInitialized;

        T *m_pos;
        T *m_vel;
        T *m_force;

        T m_softeningSquared;
        T m_damping;
};

#include "bodysystemcpu_impl.h"

#endif // __BODYSYSTEMCPU_H__
