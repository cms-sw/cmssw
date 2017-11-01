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

#ifndef NV_NPP_UTIL_STOP_WATCH_WIN_H
#define NV_NPP_UTIL_STOP_WATCH_WIN_H

// includes, system
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#undef min
#undef max

namespace npp
{

    /// Windows specific implementation of StopWatch
    class StopWatchWin
    {
        protected:

            //! Constructor, default
            StopWatchWin();

            // Destructor
            ~StopWatchWin();

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

            // member variables

            //! Start of measurement
            LARGE_INTEGER  start_time;
            //! End of measurement
            LARGE_INTEGER  end_time;

            //! Time difference between the last start and stop
            double  diff_time;

            //! TOTAL time difference between starts and stops
            double  total_time;

            //! flag if the stop watch is running
            bool running;

            //! tick frequency
            static double  freq;

            //! flag if the frequency has been set
            static  bool  freq_set;
    };

    // functions, inlined

    ////////////////////////////////////////////////////////////////////////////////
    //! Start time measurement
    ////////////////////////////////////////////////////////////////////////////////
    inline void
    StopWatchWin::start()
    {
        QueryPerformanceCounter((LARGE_INTEGER *) &start_time);
        running = true;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Stop time measurement and increment add to the current diff_time summation
    //! variable. Also increment the number of times this clock has been run.
    ////////////////////////////////////////////////////////////////////////////////
    inline void
    StopWatchWin::stop()
    {
        QueryPerformanceCounter((LARGE_INTEGER *) &end_time);
        diff_time = (float)
                    (((double) end_time.QuadPart - (double) start_time.QuadPart) / freq);

        total_time += diff_time;
        running = false;
    }

    ////////////////////////////////////////////////////////////////////////////////
    //! Reset the timer to 0. Does not change the timer running state but does
    //! recapture this point in time as the current start time if it is running.
    ////////////////////////////////////////////////////////////////////////////////
    inline void
    StopWatchWin::reset()
    {
        diff_time = 0;
        total_time = 0;

        if (running)
        {
            QueryPerformanceCounter((LARGE_INTEGER *) &start_time);
        }
    }


    ////////////////////////////////////////////////////////////////////////////////
    //! Time in msec. after start. If the stop watch is still running (i.e. there
    //! was no call to stop()) then the elapsed time is returned added to the
    //! current diff_time sum, otherwise the current summed time difference alone
    //! is returned.
    ////////////////////////////////////////////////////////////////////////////////
    inline const double
    StopWatchWin::elapsed() const
    {
        // Return the TOTAL time to date
        double retval = total_time;

        if (running)
        {
            LARGE_INTEGER temp;
            QueryPerformanceCounter((LARGE_INTEGER *) &temp);
            retval +=
                (((double)(temp.QuadPart - start_time.QuadPart)) / freq);
        }

        return retval;
    }

} // npp namespace

#endif // NV_NPP_UTIL_STOP_WATCH_WIN_H

