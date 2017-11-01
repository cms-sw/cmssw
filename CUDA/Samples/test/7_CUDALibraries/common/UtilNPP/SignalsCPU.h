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

#ifndef NV_UTIL_NPP_SIGNALS_CPU_H
#define NV_UTIL_NPP_SIGNALS_CPU_H

#include "Signal.h"

#include "SignalAllocatorsCPU.h"
#include "Exceptions.h"

#include <npp.h>


namespace npp
{

    template<typename D, class A>
    class SignalCPU: public npp::SignalTemplate<D, A>
    {
        public:
            typedef typename npp::SignalTemplate<D, A>::tData tData;

            SignalCPU()
            {
                ;
            }

            SignalCPU(size_t nSize): SignalTemplate<D, A>(nSize)
            {
                ;
            }

            SignalCPU(const SignalCPU<D, A> &rSignal): SignalTemplate<D, A>(rSignal)
            {
                ;
            }

            virtual
            ~SignalCPU()
            {
                ;
            }

            SignalCPU &
            operator= (const SignalCPU<D,A> &rSignal)
            {
                SignalTemplate<D, A>::operator= (rSignal);

                return *this;
            }

            tData &
            operator [](unsigned int i)
            {
                return *SignalTemplate<D, A>::values(i);
            }

            tData
            operator [](unsigned int i)
            const
            {
                return *SignalTemplate<D, A>::values(i);
            }

    };

    typedef SignalCPU<Npp8u,   npp::SignalAllocatorCPU<Npp8u>   >   SignalCPU_8u;
    typedef SignalCPU<Npp32s,  npp::SignalAllocatorCPU<Npp32s>  >   SignalCPU_32s;
    typedef SignalCPU<Npp16s,  npp::SignalAllocatorCPU<Npp16s>  >   SignalCPU_16s;
    typedef SignalCPU<Npp16sc, npp::SignalAllocatorCPU<Npp16sc> >   SignalCPU_16sc;
    typedef SignalCPU<Npp32sc, npp::SignalAllocatorCPU<Npp32sc> >   SignalCPU_32sc;
    typedef SignalCPU<Npp32f,  npp::SignalAllocatorCPU<Npp32f>  >   SignalCPU_32f;
    typedef SignalCPU<Npp32fc, npp::SignalAllocatorCPU<Npp32fc> >   SignalCPU_32fc;
    typedef SignalCPU<Npp64s,  npp::SignalAllocatorCPU<Npp64s>  >   SignalCPU_64s;
    typedef SignalCPU<Npp64sc, npp::SignalAllocatorCPU<Npp64sc> >   SignalCPU_64sc;
    typedef SignalCPU<Npp64f,  npp::SignalAllocatorCPU<Npp64f>  >   SignalCPU_64f;
    typedef SignalCPU<Npp64fc, npp::SignalAllocatorCPU<Npp64fc> >   SignalCPU_64fc;

} // npp namespace

#endif // NV_UTIL_NPP_SIGNALS_CPU_H
