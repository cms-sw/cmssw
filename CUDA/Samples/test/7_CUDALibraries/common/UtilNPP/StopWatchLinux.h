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

#ifndef NV_NPP_UTIL_STOP_WATCH_LINUX_H
#define NV_NPP_UTIL_STOP_WATCH_LINUX_H

// includes, system
#include <ctime>
#include <sys/time.h>

namespace npp
{

    /// Windows specific implementation of StopWatch
    class StopWatchLinux
    {

        protected:

            //! Constructor, default
            StopWatchLinux();

            // Destructor
            ~StopWatchLinux();

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

            // helper functions

            //! Get difference between start time and current time
            inline double getDiffTime() const;

        private:

            // member variables

            //! Start of measurement
            struct timeval  start_time;

            //! Time difference between the last start and stop
            double  diff_time;

            //! TOTAL time difference between starts and stops
            double  total_time;

            //! flag if the stop watch is running
            bool running;
    };

    // functions, inlined

    ////////////////////////////////////////////////////////////////////////////////
    //! Start time measurement
    ////////////////////////////////////////////////////////////////////////////////
    inline void
    StopWatchLinux::start()
    {

        gettimeofday(&start_time, 0);
        running = true;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Stop time measurement and increment add to the current diff_time summation
    //! variable. Also increment the number of times this clock has been run.
    ////////////////////////////////////////////////////////////////////////////////
    inline void
    StopWatchLinux::stop()
    {

        diff_time = getDiffTime();
        total_time += diff_time;
        running = false;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Reset the timer to 0. Does not change the timer running state but does
    //! recapture this point in time as the current start time if it is running.
    ////////////////////////////////////////////////////////////////////////////////
    inline void
    StopWatchLinux::reset()
    {
        diff_time = 0;
        total_time = 0;

        if (running)
        {
            gettimeofday(&start_time, 0);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Time in msec. after start. If the stop watch is still running (i.e. there
    //! was no call to stop()) then the elapsed time is returned added to the
    //! current diff_time sum, otherwise the current summed time difference alone
    //! is returned.
    ////////////////////////////////////////////////////////////////////////////////
    inline const double
    StopWatchLinux::elapsed() const
    {
        // Return the TOTAL time to date
        double retval = total_time;

        if (running)
        {

            retval += getDiffTime();
        }

        return retval;
    }



    ////////////////////////////////////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////
    inline double
    StopWatchLinux::getDiffTime() const
    {
        struct timeval t_time;
        gettimeofday(&t_time, 0);

        // time difference in milli-seconds
        return (1000.0 * (t_time.tv_sec - start_time.tv_sec)
                + (0.001 * (t_time.tv_usec - start_time.tv_usec)));
    }

} // npp namespace

#endif // NV_NPP_UTIL_STOP_WATCH_LINUX_H

