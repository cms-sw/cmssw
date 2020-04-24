#ifndef posix_getrusage_h
#define posix_getrusage_h

// C++ standard headers
#include <chrono>

// POSIX standard headers
#include <sys/time.h>
#include <sys/resource.h>

// based on getrusage(RUSAGE_SELF, ...)
struct clock_getrusage_self
{
  typedef std::chrono::microseconds                                     duration;
  typedef duration::rep                                                 rep;
  typedef duration::period                                              period;
  typedef std::chrono::time_point<clock_getrusage_self, duration>       time_point;

  static constexpr bool is_steady = false;

  static time_point now() noexcept
  {
    rusage ru;
    getrusage(RUSAGE_SELF, & ru);   // XXX handle possible errors

    return time_point( std::chrono::microseconds(
          ((int64_t) ru.ru_utime.tv_sec  + (int64_t) ru.ru_stime.tv_sec) * 1000000ll + 
           (int64_t) ru.ru_utime.tv_usec + (int64_t) ru.ru_stime.tv_usec) );
  }

};

#ifdef __linux__
#include <linux/version.h>
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2, 6, 26)
#define HAVE_POSIX_CLOCK_GETRUSAGE_THREAD       1

// based on getrusage(RUSAGE_THREAD, ...)
struct clock_getrusage_thread
{
  typedef std::chrono::microseconds                                     duration;
  typedef duration::rep                                                 rep;
  typedef duration::period                                              period;
  typedef std::chrono::time_point<clock_getrusage_thread, duration>     time_point;

  static constexpr bool is_steady = false;

  static time_point now() noexcept
  {
    rusage ru;
    getrusage(RUSAGE_THREAD, & ru);     // XXX handle possible errors

    return time_point( std::chrono::microseconds(
          ((int64_t) ru.ru_utime.tv_sec  + (int64_t) ru.ru_stime.tv_sec) * 1000000ll + 
           (int64_t) ru.ru_utime.tv_usec + (int64_t) ru.ru_stime.tv_usec) );
  }

};

#endif // LINUX_VERSION_CODE >= KERNEL_VERSION(2, 6, 26)
#endif // defined(__linux__)

#endif // posix_getrusage_h
