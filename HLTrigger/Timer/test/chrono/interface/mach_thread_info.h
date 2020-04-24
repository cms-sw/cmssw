#if defined(__APPLE__) || defined(__MACH__)

#ifndef mach_thread_info_h
#define mach_thread_info_h

// C++ standard headers
#include <chrono>

// Darwin system headers
#include <mach/kern_return.h>
#include <mach/thread_info.h>
#define HAVE_MACH_THREAD_INFO_CLOCK

// based on thread_info(mach_thread_self(), THREAD_BASIC_INFO, ...)
struct mach_thread_info_clock
{
  typedef std::chrono::microseconds                                     duration;
  typedef duration::rep                                                 rep;
  typedef duration::period                                              period;
  typedef std::chrono::time_point<mach_thread_info_clock, duration>     time_point;

  static constexpr bool is_steady    = false;
  static constexpr bool is_available = true;

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

#endif // mach_thread_info_h

#endif // defined(__APPLE__) || defined(__MACH__)
