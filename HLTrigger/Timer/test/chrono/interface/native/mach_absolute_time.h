#if defined(__APPLE__) || defined(__MACH__)

#ifndef native_mach_absolute_time_h
#define native_mach_absolute_time_h

// C++ standard headers
#include <chrono>

// Darwin system headers
#include <mach/mach.h>
#include <mach/mach_time.h>
#define HAVE_MACH_ABSOLUTE_TIME

#include "interface/mach_absolute_time_tick.h"

// for native_duration, etc.
#include "interface/native/native.h"

namespace native {

  // mach_absolute_time-based clock (native)
  struct mach_absolute_time_clock
  {
    // native interface
    typedef native_duration<uint64_t, mach_absolute_time_tick>                  duration;
    typedef duration::rep                                                       rep;
    typedef duration::period                                                    period;
    typedef std::chrono::time_point<mach_absolute_time_clock, duration>         time_point;

    static constexpr bool is_steady    = true;
    static constexpr bool is_available = true;

    static time_point now() noexcept
    {
      rep        ticks = mach_absolute_time();
      time_point time  = time_point(duration(ticks));
      return time;
    }

  };

} // namespace native

#endif // native_mach_absolute_time_h

#endif // defined(__APPLE__) || defined(__MACH__)
