#if defined(__APPLE__) || defined(__MACH__)

#ifndef mach_absolute_time_h
#define mach_absolute_time_h

// C++ standard headers
#include <chrono>

// Darwin system headers
#include <mach/mach.h>
#include <mach/mach_time.h>
#define HAVE_MACH_ABSOLUTE_TIME

#include "interface/mach_absolute_time_tick.h"


// mach_absolute_time-based clock
struct mach_absolute_time_clock
{
  // std::chrono interface
  typedef std::chrono::nanoseconds                                              duration;
  typedef duration::rep                                                         rep;
  typedef duration::period                                                      period;
  typedef std::chrono::time_point<mach_absolute_time_clock, duration>           time_point;

  static constexpr bool is_steady    = true;
  static constexpr bool is_available = true;

  static time_point now() noexcept
  {
    uint64_t   ticks  = mach_absolute_time();
    rep        ns     = mach_absolute_time_tick::to_nanoseconds(ticks);
    time_point time   = time_point(duration(ns));
    return time;
  }
};


#endif // mach_absolute_time_h

#endif // defined(__APPLE__) || defined(__MACH__)
