#include "interface/syscall_clock_gettime.h"

#ifdef HAVE_SYSCALL_CLOCK_REALTIME
const bool clock_syscall_realtime::is_available = true;
#endif // HAVE_SYSCALL_CLOCK_REALTIME


#ifdef HAVE_SYSCALL_CLOCK_REALTIME_COARSE
const bool clock_syscall_realtime_coarse::is_available = true;
#endif // HAVE_SYSCALL_CLOCK_REALTIME_COARSE


#ifdef HAVE_SYSCALL_CLOCK_MONOTONIC
const bool clock_syscall_monotonic::is_available = true;
#endif // HAVE_SYSCALL_CLOCK_MONOTONIC


#ifdef HAVE_SYSCALL_CLOCK_MONOTONIC_COARSE
const bool clock_syscall_monotonic_coarse::is_available = true;
#endif // HAVE_SYSCALL_CLOCK_MONOTONIC_COARSE


#ifdef HAVE_SYSCALL_CLOCK_MONOTONIC_RAW
const bool clock_syscall_monotonic_raw::is_available = true;
#endif // HAVE_SYSCALL_CLOCK_MONOTONIC_RAW


#ifdef HAVE_SYSCALL_CLOCK_BOOTTIME
const bool clock_syscall_boottime::is_available = true;
#endif // HAVE_SYSCALL_CLOCK_BOOTTIME


#ifdef HAVE_SYSCALL_CLOCK_PROCESS_CPUTIME_ID
const bool clock_syscall_process_cputime::is_available = true;
#endif // HAVE_SYSCALL_CLOCK_PROCESS_CPUTIME_ID


#ifdef HAVE_SYSCALL_CLOCK_THREAD_CPUTIME_ID
const bool clock_syscall_thread_cputime::is_available = true;
#endif // HAVE_SYSCALL_CLOCK_THREAD_CPUTIME_ID
