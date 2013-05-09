#if defined(__APPLE__) || defined(__MACH__)

#ifndef mach_absolute_time_h
#define mach_absolute_time_h

// C++ standard headers
#include <chrono>

// Darwin system headers
#include <mach/mach.h>
#include <mach/mach_time.h>
#define HAVE_MACH_ABSOLUTE_TIME

// for native_duration, etc.
#include "native.h"


// mach_absolute_time ticks as clock period
// XXX should it use unsigned integers ?
struct mach_absolute_time_tick {
  static const double  ticks_per_second;
  static const double  seconds_per_tick;
  static const int64_t nanoseconds_per_tick_shifted;
  static const int64_t ticks_per_nanosecond_shifted;

  static int64_t to_nanoseconds(int64_t ticks) noexcept
  {
    // round the shifted value away from 0, like round() does
    // XXX should it honor fesetround instead ?
    __int128_t shifted = (__int128_t) ticks * nanoseconds_per_tick_shifted;
    __int128_t ns = (shifted >> 32) + ((shifted & 0xffffffff) >= 0x80000000);
    return (int64_t) ns;
  }

  static double to_seconds(double ticks) noexcept
  {
    return ticks / ticks_per_second;
  }

  static int64_t from_nanoseconds(int64_t ns) noexcept {
    // round the shifted value away from 0, like round() does
    // XXX should it honor fesetround instead ?
    __int128_t shifted = (__int128_t) ns * ticks_per_nanosecond_shifted;
    __int128_t ticks = (shifted >> 32) + ((shifted & 0xffffffff) >= 0x80000000);
    return (int64_t) ticks;
  }

  static int64_t from_seconds(double seconds) noexcept {
    // XXX use lrint intead of lround (honors fesetround) ?
    return (int64_t) std::lround(seconds * ticks_per_second);
  }

  template <typename _ToRep, typename _ToPeriod>
  static
  typename std::enable_if<
    std::chrono::treat_as_floating_point<_ToRep>::value,
    std::chrono::duration<_ToRep, _ToPeriod>>::type
  to_duration(double ticks)
  {
    std::chrono::duration<double> d(to_seconds(ticks));
    return std::chrono::duration_cast<std::chrono::duration<_ToRep, _ToPeriod>>( d );
  }

  template <typename _ToRep, typename _ToPeriod>
  static
  typename std::enable_if<
    not std::chrono::treat_as_floating_point<_ToRep>::value,
    std::chrono::duration<_ToRep, _ToPeriod>>::type
  to_duration(int64_t ticks)
  {
    std::chrono::nanoseconds d(to_nanoseconds(ticks));
    return std::chrono::duration_cast<std::chrono::duration<_ToRep, _ToPeriod>>( d );
  }

  template <typename _FromRep, typename _FromPeriod>
  static
  typename std::enable_if<
    std::chrono::treat_as_floating_point<_FromRep>::value,
    double>::type
  from_duration(std::chrono::duration<_FromRep, _FromPeriod> d)
  {
    double s = std::chrono::duration_cast<std::chrono::duration<double>>(d).count();
    return from_seconds(s);
  }

  template <typename _FromRep, typename _FromPeriod>
  static
  typename std::enable_if<
    not std::chrono::treat_as_floating_point<_FromRep>::value,
    int64_t>::type
  from_duration(std::chrono::duration<_FromRep, _FromPeriod> d)
  {
    int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(d).count();
    return from_nanoseconds(ns);
  }

};


// mach_absolute_time ticks as clock duration
typedef native_duration<uint64_t, mach_absolute_time_tick> mach_absolute_time_duration;


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
    return from_native(native_now());
  }


  // native interface
  typedef mach_absolute_time_duration                                           native_duration;
  typedef native_duration::rep                                                  native_rep;
  typedef native_duration::period                                               native_period;
  typedef std::chrono::time_point<mach_absolute_time_clock, native_duration>    native_time_point;

  static native_time_point native_now() noexcept
  {
    native_rep        ticks = mach_absolute_time();
    native_duration   d(ticks);
    native_time_point t(d);
    return t;
  }

  static time_point from_native(const native_time_point & native_time) noexcept
  {
    native_rep native = native_time.time_since_epoch().count();
    rep        ns     = native_period::to_nanoseconds(native);
    time_point time   = time_point(duration(ns));
    return time;
  }
};

#endif // mach_absolute_time_h

#endif // defined(__APPLE__) || defined(__MACH__)
