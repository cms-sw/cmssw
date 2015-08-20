#ifndef thread_clock_h
#define thread_clock_h

// POSIX standard headers
#include <unistd.h>

// check POSIX capabilities
#if (defined(_POSIX_TIMERS) && (_POSIX_TIMERS >= 0))
#if (defined(_POSIX_THREAD_CPUTIME) && (_POSIX_THREAD_CPUTIME >= 0))

// POSIX standard headers
#include <time.h>
#define HAVE_POSIX_CLOCK_THREAD_CPUTIME_ID

#endif // _POSIX_THREAD_CPUTIME
#endif // _POSIX_TIMERS

// check for OS X
#if ! defined(HAVE_POSIX_CLOCK_THREAD_CPUTIME_ID)
#if defined(__APPLE__) || defined(__MACH__)

// Darwin system headers
#include <mach/kern_return.h>
#include <mach/thread_info.h>
#define HAVE_MACH_THREAD_INFO_CLOCK

#endif // defined(__APPLE__) || defined(__MACH__)
#endif // ! defined(HAVE_POSIX_CLOCK_THREAD_CPUTIME_ID)

// C++ standard headers
#include <chrono>

namespace cms {

namespace chrono {

#if defined(HAVE_POSIX_CLOCK_THREAD_CPUTIME_ID)

// per-thread clock based on clock_gettime(CLOCK_THREAD_CPUTIME_ID, ...)
struct thread_clock
{
  typedef std::chrono::nanoseconds                          duration;
  typedef duration::rep                                     rep;
  typedef duration::period                                  period;
  typedef std::chrono::time_point<thread_clock, duration>   time_point;

  static constexpr bool is_steady = false;

  static time_point now() noexcept
  {
    timespec t;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);

    return time_point( std::chrono::seconds(t.tv_sec) + std::chrono::nanoseconds(t.tv_nsec) );
  }

};

#elif defined(HAVE_MACH_THREAD_INFO_CLOCK)

// per-thread clock based on thread_info(mach_thread_self(), THREAD_BASIC_INFO, ...)
struct thread_clock
{
  typedef std::chrono::microseconds                         duration;
  typedef duration::rep                                     rep;
  typedef duration::period                                  period;
  typedef std::chrono::time_point<thread_clock, duration>   time_point;

  static constexpr bool is_steady = false;

  static time_point now() noexcept
  {
    thread_local mach_port_t thread_id = mach_thread_self();

    thread_basic_info_data_t basic_info;
    mach_msg_type_number_t count = THREAD_BASIC_INFO_COUNT;
    if (KERN_SUCCESS == thread_info(thread_id, THREAD_BASIC_INFO, (thread_info_t) & basic_info, & count)) {
      return time_point( std::chrono::seconds(basic_info.user_time.seconds + basic_info.system_time.seconds) +
                         std::chrono::microseconds(basic_info.user_time.microseconds + basic_info.system_time.microseconds) );
    } else {
      return time_point();
    }
  }

};

#else

#warning "no underlying per-thread clock is available, thread_clock::now() will return 0."

// no per-thread clock is available
struct thread_clock
{
  typedef std::chrono::seconds                              duration;
  typedef duration::rep                                     rep;
  typedef duration::period                                  period;
  typedef std::chrono::time_point<thread_clock, duration>   time_point;

  static constexpr bool is_steady = false;

  static time_point now() noexcept
  {
    return time_point();
  }

};

#endif

}

}

#endif // thread_clock_h
