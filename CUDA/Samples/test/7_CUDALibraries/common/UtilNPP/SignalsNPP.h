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

#ifndef NV_UTIL_NPP_SIGNALS_NPP_H
#define NV_UTIL_NPP_SIGNALS_NPP_H

#include "Exceptions.h"
#include "Signal.h"

#include "SignalAllocatorsNPP.h"
#include <cuda_runtime.h>

namespace npp
{
    // forward declaration
    template<typename D, class A> class SignalCPU;

    template<typename D>
    class SignalNPP: public npp::SignalTemplate<D, npp::SignalAllocator<D> >
    {
        public:
            SignalNPP()
            {
                ;
            }

            explicit
            SignalNPP(size_t nSize): SignalTemplate<D, npp::SignalAllocator<D> >(nSize)
            {
                ;
            }

            SignalNPP(const SignalNPP<D> &rSignal): SignalTemplate<D, npp::SignalAllocator<D> >(rSignal)
            {
                ;
            }

            template<class X>
            explicit
            SignalNPP(const SignalCPU<D, X> &rSignal): SignalTemplate<D, npp::SignalAllocator<D> >(rSignal.size())
            {
                npp::SignalAllocator<D>::HostToDeviceCopy1D(SignalTemplate<D, npp::SignalAllocator<D> >::values(),
                                                            rSignal.values(), SignalTemplate<D, npp::SignalAllocator<D> >::size());
            }

            virtual
            ~SignalNPP()
            {
                ;
            }

            SignalNPP &
            operator= (const SignalNPP<D> &rSignal)
            {
                SignalTemplate<D, npp::SignalAllocator<D> >::operator= (rSignal);

                return *this;
            }

            void
            copyTo(D *pValues)
            const
            {
                npp::SignalAllocator<D>::DeviceToHostCopy1D(pValues, SignalTemplate<D, npp::SignalAllocator<D> >::values(), SignalTemplate<D, npp::SignalAllocator<D> >::size());
            }

            void
            copyFrom(D *pValues)
            {
                npp::SignalAllocator<D>::HostToDeviceCopy1D(SignalTemplate<D, npp::SignalAllocator<D> >::values(), pValues, SignalTemplate<D, npp::SignalAllocator<D> >::size());
            }
    };

    typedef SignalNPP<Npp8u>    SignalNPP_8u;
    typedef SignalNPP<Npp16s>   SignalNPP_16s;
    typedef SignalNPP<Npp16sc>  SignalNPP_16sc;
    typedef SignalNPP<Npp32s>   SignalNPP_32s;
    typedef SignalNPP<Npp32sc>  SignalNPP_32sc;
    typedef SignalNPP<Npp32f>   SignalNPP_32f;
    typedef SignalNPP<Npp32fc>  SignalNPP_32fc;
    typedef SignalNPP<Npp64s>   SignalNPP_64s;
    typedef SignalNPP<Npp64sc>  SignalNPP_64sc;
    typedef SignalNPP<Npp64f>   SignalNPP_64f;
    typedef SignalNPP<Npp64fc>  SignalNPP_64fc;

} // npp namespace

#endif // NV_UTIL_NPP_SIGNALS_NPP_H
