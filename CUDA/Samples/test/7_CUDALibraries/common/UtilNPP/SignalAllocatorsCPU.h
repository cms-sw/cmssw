/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef NV_UTIL_NPP_SIGNAL_ALLOCATORS_CPU_H
#define NV_UTIL_NPP_SIGNAL_ALLOCATORS_CPU_H

#include "Exceptions.h"

namespace npp
{

    template <typename D>
    class SignalAllocatorCPU
    {
        public:
            static
            D *
            Malloc1D(unsigned int nSize)
            {
                return new D[nSize];;
            };

            static
            void
            Free1D(D *pPixels)
            {
                delete[] pPixels;
            };

            static
            void
            Copy1D(D *pDst, const D *pSrc, size_t nSize)
            {
                memcpy(pDst, pSrc, nSize * sizeof(D));
            };

    };

} // npp namespace

#endif // NV_UTIL_NPP_SIGNAL_ALLOCATORS_CPU_H
