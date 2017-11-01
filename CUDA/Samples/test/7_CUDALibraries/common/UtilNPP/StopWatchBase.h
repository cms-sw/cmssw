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

#ifndef NV_NPP_UTIL_STOP_WATCH_BASE_H
#define NV_NPP_UTIL_STOP_WATCH_BASE_H


namespace npp
{

    /// Simple stop watch class
    ///     This class uses high-precision timers. It is implemented
    /// using templates and inline functions to cause minimal call-overhead
    /// and provide the most accurate timings.
    template<class OSPolicy>
    class StopWatchBase : public OSPolicy
    {
        public:

            // generic, specialized type
            typedef StopWatchBase<OSPolicy>   SelfType;
            // generic, specialized type pointer
            typedef StopWatchBase<OSPolicy>  *SelfTypePtr;

        public:

            //! Constructor, default
            StopWatchBase();

            // Destructor
            ~StopWatchBase();

        public:

            //! Start time measurement
            inline void start();

            //! Stop time measurement
            inline void stop();

            //! Reset time counters to zero
            inline void reset();

            //! Time in msec. after start. If the stop watch is still running (i.e. there
            //! was no call to stop()) then the elapsed time is returned, otherwise the
            //! time between the last start() and stop call is returned
            inline const double elapsed() const;

        private:

            //! Constructor, copy (not implemented)
            StopWatchBase(const StopWatchBase &);

            //! Assignment operator (not implemented)
            StopWatchBase &operator=(const StopWatchBase &);
    };

    // include, implementation
#include "StopWatchBase.inl"

} // npp namespace

#endif // NV_NPP_UTIL_STOP_WATCH_BASE_H

