#ifndef syscall_clock_gettime_h
#define syscall_clock_gettime_h

// C++ standard headers
#include <chrono>

// POSIX standard headers
#include <unistd.h>
#include <time.h>

// check available capabilities
#ifdef __linux__
#include <linux/version.h>
#include <sys/syscall.h>

#define HAVE_SYSCALL_CLOCK_REALTIME
#define HAVE_SYSCALL_CLOCK_MONOTONIC           
#define HAVE_SYSCALL_CLOCK_PROCESS_CPUTIME_ID
#define HAVE_SYSCALL_CLOCK_THREAD_CPUTIME_ID

#if LINUX_VERSION_CODE >= KERNEL_VERSION(2, 6, 28)
#define HAVE_SYSCALL_CLOCK_MONOTONIC_RAW
#endif // LINUX_VERSION_CODE >= KERNEL_VERSION(2, 6, 28)
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2, 6, 32)
#define HAVE_SYSCALL_CLOCK_REALTIME_COARSE
#define HAVE_SYSCALL_CLOCK_MONOTONIC_COARSE
#endif // LINUX_VERSION_CODE >= KERNEL_VERSION(2, 6, 32)
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2, 6, 39)
#define HAVE_SYSCALL_CLOCK_BOOTTIME
#endif // LINUX_VERSION_CODE >= KERNEL_VERSION(2, 6, 39)

#ifdef HAVE_SYSCALL_CLOCK_REALTIME
// based on syscall(SYS_clock_gettime, CLOCK_REALTIME, ...)
struct clock_syscall_realtime
{
  typedef std::chrono::nanoseconds                                      duration;
  typedef duration::rep                                                 rep;
  typedef duration::period                                              period;
  typedef std::chrono::time_point<clock_syscall_realtime, duration>     time_point;

  static constexpr bool is_steady = false;
  static const     bool is_available;

  static time_point now() noexcept
  {
    timespec t;
    syscall(SYS_clock_gettime, CLOCK_REALTIME, &t);

    return time_point( std::chrono::seconds(t.tv_sec) + std::chrono::nanoseconds(t.tv_nsec) );
  }

};
#endif // HAVE_SYSCALL_CLOCK_REALTIME


#ifdef HAVE_SYSCALL_CLOCK_REALTIME_COARSE
// based on syscall(SYS_clock_gettime, CLOCK_REALTIME_COARSE, ...)
struct clock_syscall_realtime_coarse
{
  typedef std::chrono::nanoseconds                                          duration;
  typedef duration::rep                                                     rep;
  typedef duration::period                                                  period;
  typedef std::chrono::time_point<clock_syscall_realtime_coarse, duration>  time_point;

  static constexpr bool is_steady = false;
  static const     bool is_available;

  static time_point now() noexcept
  {
    timespec t;
    syscall(SYS_clock_gettime, CLOCK_REALTIME_COARSE, &t);

    return time_point( std::chrono::seconds(t.tv_sec) + std::chrono::nanoseconds(t.tv_nsec) );
  }

};
#endif // HAVE_SYSCALL_CLOCK_REALTIME_COARSE


#ifdef HAVE_SYSCALL_CLOCK_MONOTONIC
// based on syscall(SYS_clock_gettime, CLOCK_MONOTONIC, ...)
struct clock_syscall_monotonic
{
  typedef std::chrono::nanoseconds                                      duration;
  typedef duration::rep                                                 rep;
  typedef duration::period                                              period;
  typedef std::chrono::time_point<clock_syscall_monotonic, duration>    time_point;

  static constexpr bool is_steady = true;
  static const     bool is_available;

  static time_point now() noexcept
  {
    timespec t;
    syscall(SYS_clock_gettime, CLOCK_MONOTONIC, &t);

    return time_point( std::chrono::seconds(t.tv_sec) + std::chrono::nanoseconds(t.tv_nsec) );
  }

};
#endif // HAVE_SYSCALL_CLOCK_MONOTONIC


#ifdef HAVE_SYSCALL_CLOCK_MONOTONIC_COARSE
// based on syscall(SYS_clock_gettime, CLOCK_MONOTONIC_COARSE, ...)
struct clock_syscall_monotonic_coarse
{
  typedef std::chrono::nanoseconds                                          duration;
  typedef duration::rep                                                     rep;
  typedef duration::period                                                  period;
  typedef std::chrono::time_point<clock_syscall_monotonic_coarse, duration> time_point;

  static constexpr bool is_steady = true;
  static const     bool is_available;

  static time_point now() noexcept
  {
    timespec t;
    syscall(SYS_clock_gettime, CLOCK_MONOTONIC_COARSE, &t);

    return time_point( std::chrono::seconds(t.tv_sec) + std::chrono::nanoseconds(t.tv_nsec) );
  }

};
#endif // HAVE_SYSCALL_CLOCK_MONOTONIC_COARSE


#ifdef HAVE_SYSCALL_CLOCK_MONOTONIC_RAW
// based on syscall(SYS_clock_gettime, CLOCK_MONOTONIC_RAW, ...)
struct clock_syscall_monotonic_raw
{
  typedef std::chrono::nanoseconds                                          duration;
  typedef duration::rep                                                     rep;
  typedef duration::period                                                  period;
  typedef std::chrono::time_point<clock_syscall_monotonic_raw, duration>    time_point;

  static constexpr bool is_steady = true;
  static const     bool is_available;

  static inline time_point now() noexcept
  {
    timespec t;
    syscall(SYS_clock_gettime, CLOCK_MONOTONIC_RAW, &t);

    return time_point( std::chrono::seconds(t.tv_sec) + std::chrono::nanoseconds(t.tv_nsec) );
  }

};
#endif // HAVE_SYSCALL_CLOCK_MONOTONIC_RAW


#ifdef HAVE_SYSCALL_CLOCK_BOOTTIME
// based on syscall(SYS_clock_gettime, CLOCK_BOOTTIME, ...)
struct clock_syscall_boottime
{
  typedef std::chrono::nanoseconds                                          duration;
  typedef duration::rep                                                     rep;
  typedef duration::period                                                  period;
  typedef std::chrono::time_point<clock_syscall_boottime, duration>         time_point;

  static constexpr bool is_steady = true;
  static const     bool is_available;

  static inline time_point now() noexcept
  {
    timespec t;
    syscall(SYS_clock_gettime, CLOCK_BOOTTIME, &t);

    return time_point( std::chrono::seconds(t.tv_sec) + std::chrono::nanoseconds(t.tv_nsec) );
  }

};
#endif // HAVE_SYSCALL_CLOCK_BOOTTIME


#ifdef HAVE_SYSCALL_CLOCK_PROCESS_CPUTIME_ID
// based on syscall(SYS_clock_gettime, CLOCK_PROCESS_CPUTIME_ID, ...)
struct clock_syscall_process_cputime
{
  typedef std::chrono::nanoseconds                                              duration;
  typedef duration::rep                                                         rep;
  typedef duration::period                                                      period;
  typedef std::chrono::time_point<clock_syscall_process_cputime, duration>      time_point;

  static constexpr bool is_steady = false;  // FIXME can this be considered "steady" ?
  static const     bool is_available;

  static time_point now() noexcept
  {
    timespec t;
    syscall(SYS_clock_gettime, CLOCK_PROCESS_CPUTIME_ID, &t);

    return time_point( std::chrono::seconds(t.tv_sec) + std::chrono::nanoseconds(t.tv_nsec) );
  }

};
#endif // HAVE_SYSCALL_CLOCK_PROCESS_CPUTIME_ID


#ifdef HAVE_SYSCALL_CLOCK_THREAD_CPUTIME_ID
// based on syscall(SYS_clock_gettime, CLOCK_THREAD_CPUTIME_ID, ...)
struct clock_syscall_thread_cputime
{
  typedef std::chrono::nanoseconds                                              duration;
  typedef duration::rep                                                         rep;
  typedef duration::period                                                      period;
  typedef std::chrono::time_point<clock_syscall_thread_cputime, duration>       time_point;

  static constexpr bool is_steady = false;  // FIXME can this be considered "steady" ?
  static const     bool is_available;

  static time_point now() noexcept
  {
    timespec t;
    syscall(SYS_clock_gettime, CLOCK_THREAD_CPUTIME_ID, &t);

    return time_point( std::chrono::seconds(t.tv_sec) + std::chrono::nanoseconds(t.tv_nsec) );
  }

};
#endif // HAVE_SYSCALL_CLOCK_THREAD_CPUTIME_ID

#endif // __linux__

#endif // syscall_clock_gettime_h
