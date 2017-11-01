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

#ifndef NV_UTIL_NPP_SIGNAL_H
#define NV_UTIL_NPP_SIGNAL_H

#include <cstring>

namespace npp
{
    class Signal
    {
        public:
            Signal() : nSize_(0)
            { };

            explicit
            Signal(size_t nSize) : nSize_(nSize)
            { };

            Signal(const Signal &rSignal) : nSize_(rSignal.nSize_)
            { };

            virtual
            ~Signal()
            { }

            Signal &
            operator= (const Signal &rSignal)
            {
                nSize_ = rSignal.nSize_;
                return *this;
            }

            size_t
            size()
            const
            {
                return nSize_;
            }

            void
            swap(Signal &rSignal)
            {
                size_t nTemp = nSize_;
                nSize_ = rSignal.nSize_;
                rSignal.nSize_ = nTemp;
            }


        private:
            size_t nSize_;
    };

    template<typename D, class A>
    class SignalTemplate: public Signal
    {
        public:
            typedef D tData;

            SignalTemplate(): aValues_(0)
            {
                ;
            }

            SignalTemplate(size_t nSize): Signal(nSize)
                , aValues_(0)
            {
                aValues_ = A::Malloc1D(size());
            }

            SignalTemplate(const SignalTemplate<D, A> &rSignal): Signal(rSignal)
                , aValues_(0)
            {
                aValues_ = A::Malloc1D(size());
                A::Copy1D(aValues_, rSignal.values(), size());
            }

            virtual
            ~SignalTemplate()
            {
                A::Free1D(aValues_);
            }

            SignalTemplate &
            operator= (const SignalTemplate<D, A> &rSignal)
            {
                // in case of self-assignment
                if (&rSignal == this)
                {
                    return *this;
                }

                A::Free1D(aValues_);
                this->aPixels_ = 0;

                // assign parent class's data fields (width, height)
                Signal::operator =(rSignal);

                aValues_ = A::Malloc1D(size());
                A::Copy1D(aValues_, rSignal.value(), size());

                return *this;
            }

            /// Get a pointer to the pixel array.
            ///     The result pointer can be offset to pixel at position (x, y) and
            /// even negative offsets are allowed.
            /// \param nX Horizontal pointer/array offset.
            /// \param nY Vertical pointer/array offset.
            /// \return Pointer to the pixel array (or first pixel in array with coordinates (nX, nY).
            tData *
            values(int i = 0)
            {
                return aValues_ + i;
            }

            const
            tData *
            values(int i = 0)
            const
            {
                return aValues_ + i;
            }

            void
            swap(SignalTemplate<D, A> &rSignal)
            {
                Signal::swap(rSignal);

                tData *aTemp       = this->aValues_;
                this->aValues_      = rSignal.aValues_;
                rSignal.aValues_    = aTemp;
            }

        private:
            D *aValues_;
    };

} // npp namespace


#endif // NV_UTIL_NPP_SIGNAL_H
