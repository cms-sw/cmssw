#include "interface/posix_clock_gettime.h"

#include <unistd.h>

#ifdef HAVE_POSIX_CLOCK_REALTIME
const bool clock_gettime_realtime::is_available = (sysconf(_SC_TIMERS) > 0);
#endif // HAVE_POSIX_CLOCK_REALTIME


#ifdef HAVE_POSIX_CLOCK_REALTIME_COARSE
const bool clock_gettime_realtime_coarse::is_available = true;
#endif // HAVE_POSIX_CLOCK_REALTIME_COARSE


#ifdef HAVE_POSIX_CLOCK_MONOTONIC
const bool clock_gettime_monotonic::is_available = (sysconf(_SC_MONOTONIC_CLOCK) > 0);
#endif // HAVE_POSIX_CLOCK_MONOTONIC


#ifdef HAVE_POSIX_CLOCK_MONOTONIC_COARSE
const bool clock_gettime_monotonic_coarse::is_available = true;
#endif // HAVE_POSIX_CLOCK_MONOTONIC_COARSE


#ifdef HAVE_POSIX_CLOCK_MONOTONIC_RAW
const bool clock_gettime_monotonic_raw::is_available = true;
#endif // HAVE_POSIX_CLOCK_MONOTONIC_RAW


#ifdef HAVE_POSIX_CLOCK_PROCESS_CPUTIME_ID
const bool clock_gettime_process_cputime::is_available = (sysconf(_SC_CPUTIME) > 0);
#endif // HAVE_POSIX_CLOCK_PROCESS_CPUTIME_ID


#ifdef HAVE_POSIX_CLOCK_THREAD_CPUTIME_ID
const bool clock_gettime_thread_cputime::is_available = (sysconf(_SC_THREAD_CPUTIME) > 0);
#endif // HAVE_POSIX_CLOCK_THREAD_CPUTIME_ID
