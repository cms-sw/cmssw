#ifndef posix_clock_gettime_h
#define posix_clock_gettime_h

// C++ standard headers
#include <chrono>

// POSIX standard headers
#include <unistd.h>
#include <time.h>

// check available capabilities
#if (defined(_POSIX_TIMERS) && (_POSIX_TIMERS >= 0))

#define HAVE_POSIX_CLOCK_REALTIME

#if (defined(_POSIX_MONOTONIC_CLOCK) && (_POSIX_MONOTONIC_CLOCK >= 0))
#define HAVE_POSIX_CLOCK_MONOTONIC           
#endif // _POSIX_MONOTONIC_CLOCK

#if (defined(_POSIX_CPUTIME) && (_POSIX_CPUTIME >= 0))
#define HAVE_POSIX_CLOCK_PROCESS_CPUTIME_ID
#endif // _POSIX_CPUTIME

#if (defined(_POSIX_THREAD_CPUTIME) && (_POSIX_THREAD_CPUTIME >= 0))
#define HAVE_POSIX_CLOCK_THREAD_CPUTIME_ID
#endif // _POSIX_THREAD_CPUTIME

#ifdef __linux__
#include <linux/version.h>
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2, 6, 28)
#define HAVE_POSIX_CLOCK_MONOTONIC_RAW
#endif // LINUX_VERSION_CODE >= KERNEL_VERSION(2, 6, 28)
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2, 6, 32)
#define HAVE_POSIX_CLOCK_REALTIME_COARSE
#define HAVE_POSIX_CLOCK_MONOTONIC_COARSE
#endif // LINUX_VERSION_CODE >= KERNEL_VERSION(2, 6, 32)
#endif // __linux__

#endif // _POSIX_TIMERS

#ifdef HAVE_POSIX_CLOCK_REALTIME
// based on clock_gettime(CLOCK_REALTIME, ...)
struct clock_gettime_realtime
{
  typedef std::chrono::nanoseconds                                      duration;
  typedef duration::rep                                                 rep;
  typedef duration::period                                              period;
  typedef std::chrono::time_point<clock_gettime_realtime, duration>     time_point;

  static constexpr bool is_steady = false;
  static const     bool is_available;

  static time_point now() noexcept
  {
    timespec t;
    clock_gettime(CLOCK_REALTIME, &t);

    return time_point( std::chrono::seconds(t.tv_sec) + std::chrono::nanoseconds(t.tv_nsec) );
  }

};
#endif // HAVE_POSIX_CLOCK_REALTIME


#ifdef HAVE_POSIX_CLOCK_REALTIME_COARSE
// based on clock_gettime(CLOCK_REALTIME_COARSE, ...)
struct clock_gettime_realtime_coarse
{
  typedef std::chrono::nanoseconds                                          duration;
  typedef duration::rep                                                     rep;
  typedef duration::period                                                  period;
  typedef std::chrono::time_point<clock_gettime_realtime_coarse, duration>  time_point;

  static constexpr bool is_steady = false;
  static const     bool is_available;

  static time_point now() noexcept
  {
    timespec t;
    clock_gettime(CLOCK_REALTIME_COARSE, &t);

    return time_point( std::chrono::seconds(t.tv_sec) + std::chrono::nanoseconds(t.tv_nsec) );
  }

};
#endif // HAVE_POSIX_CLOCK_REALTIME_COARSE


#ifdef HAVE_POSIX_CLOCK_MONOTONIC
// based on clock_gettime(CLOCK_MONOTONIC, ...)
struct clock_gettime_monotonic
{
  typedef std::chrono::nanoseconds                                      duration;
  typedef duration::rep                                                 rep;
  typedef duration::period                                              period;
  typedef std::chrono::time_point<clock_gettime_monotonic, duration>    time_point;

  static constexpr bool is_steady = true;
  static const     bool is_available;

  static time_point now() noexcept
  {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);

    return time_point( std::chrono::seconds(t.tv_sec) + std::chrono::nanoseconds(t.tv_nsec) );
  }

};
#endif // HAVE_POSIX_CLOCK_MONOTONIC


#ifdef HAVE_POSIX_CLOCK_MONOTONIC_COARSE
// based on clock_gettime(CLOCK_MONOTONIC_COARSE, ...)
struct clock_gettime_monotonic_coarse
{
  typedef std::chrono::nanoseconds                                          duration;
  typedef duration::rep                                                     rep;
  typedef duration::period                                                  period;
  typedef std::chrono::time_point<clock_gettime_monotonic_coarse, duration> time_point;

  static constexpr bool is_steady = true;
  static const     bool is_available;

  static time_point now() noexcept
  {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC_COARSE, &t);

    return time_point( std::chrono::seconds(t.tv_sec) + std::chrono::nanoseconds(t.tv_nsec) );
  }

};
#endif // HAVE_POSIX_CLOCK_MONOTONIC_COARSE


#ifdef HAVE_POSIX_CLOCK_MONOTONIC_RAW
// based on clock_gettime(CLOCK_MONOTONIC_RAW, ...)
struct clock_gettime_monotonic_raw
{
  typedef std::chrono::nanoseconds                                          duration;
  typedef duration::rep                                                     rep;
  typedef duration::period                                                  period;
  typedef std::chrono::time_point<clock_gettime_monotonic_raw, duration>    time_point;

  static constexpr bool is_steady = true;
  static const     bool is_available;

  static inline time_point now() noexcept
  {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);

    return time_point( std::chrono::seconds(t.tv_sec) + std::chrono::nanoseconds(t.tv_nsec) );
  }

};
#endif // HAVE_POSIX_CLOCK_MONOTONIC_RAW


#ifdef HAVE_POSIX_CLOCK_PROCESS_CPUTIME_ID
// based on clock_gettime(CLOCK_PROCESS_CPUTIME_ID, ...)
struct clock_gettime_process_cputime
{
  typedef std::chrono::nanoseconds                                              duration;
  typedef duration::rep                                                         rep;
  typedef duration::period                                                      period;
  typedef std::chrono::time_point<clock_gettime_process_cputime, duration>      time_point;

  static constexpr bool is_steady = false;  // FIXME can this be considered "steady" ?
  static const     bool is_available;

  static time_point now() noexcept
  {
    timespec t;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);

    return time_point( std::chrono::seconds(t.tv_sec) + std::chrono::nanoseconds(t.tv_nsec) );
  }

};
#endif // HAVE_POSIX_CLOCK_PROCESS_CPUTIME_ID


#ifdef HAVE_POSIX_CLOCK_THREAD_CPUTIME_ID
// based on clock_gettime(CLOCK_THREAD_CPUTIME_ID, ...)
struct clock_gettime_thread_cputime
{
  typedef std::chrono::nanoseconds                                              duration;
  typedef duration::rep                                                         rep;
  typedef duration::period                                                      period;
  typedef std::chrono::time_point<clock_gettime_thread_cputime, duration>       time_point;

  static constexpr bool is_steady = false;  // FIXME can this be considered "steady" ?
  static const     bool is_available;

  static time_point now() noexcept
  {
    timespec t;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);

    return time_point( std::chrono::seconds(t.tv_sec) + std::chrono::nanoseconds(t.tv_nsec) );
  }

};
#endif // HAVE_POSIX_CLOCK_THREAD_CPUTIME_ID

#endif // posix_clock_gettime_h
