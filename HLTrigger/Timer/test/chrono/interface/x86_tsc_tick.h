#if defined __x86_64__ or defined __i386__
// TSC is only available on x86

#ifndef x86_tsc_tick_h
#define x86_tsc_tick_h

// C++ standard headers
#include <chrono>
#include <cmath>


// TSC ticks as clock period
// XXX should it use unsigned integers ?
struct tsc_tick {
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


#endif // x86_tsc_tick_h

#endif // defined __x86_64__ or defined __i386__
