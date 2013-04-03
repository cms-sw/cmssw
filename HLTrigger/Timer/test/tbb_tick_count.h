#ifndef tbb_tick_count_h
#define tbb_tick_count_h

// C++ standard headers
#include <chrono>

// TBB headers
#include <tbb/tick_count.h>

/* 
// for native_duration, etc.
#include "native.h"

// TBB ticks as clock period
struct tbb_tick {

  template <typename _ToRep, typename _ToPeriod>
  static
  std::chrono::duration<_ToRep, _ToPeriod>
  to_duration(tbb::tick_count::interval_t ticks)
  {
    std::chrono::duration<double> d(ticks.seconds());
    return std::chrono::duration_cast<std::chrono::duration<_ToRep, _ToPeriod>>( d );
  }

  template <typename _FromRep, typename _FromPeriod>
  static
  tbb::tick_count::interval_t
  from_duration(std::chrono::duration<_FromRep, _FromPeriod> d)
  {
    double s = std::chrono::duration_cast<std::chrono::duration<double>>(d).count();
    return tbb::tick_count::interval_t(s);
  }

};


// TSC ticks as clock duration
typedef native_duration<tbb::tick_count::interval_t, tbb_tick> tbb_duration;
*/


// TBB tick_count based clock
struct clock_tbb_tick_count
{
  // std::chrono interface
  typedef std::chrono::duration<double>                                     duration;
  typedef duration::rep                                                     rep;
  typedef duration::period                                                  period;
  typedef std::chrono::time_point<clock_tbb_tick_count, duration>           time_point;

  static constexpr bool is_steady = false;

  static const tbb::tick_count epoch;

  static time_point now() noexcept
  {
    // TBB tick_count does not expose its internal representation, only the the conversion of intervals to and from seconds
    tbb::tick_count             time  = tbb::tick_count::now();
    tbb::tick_count::interval_t ticks = time - epoch;
    return time_point(duration( ticks.seconds() ));
  }

  /*
  // native interface
  typedef tbb_duration                                                      native_duration;
  typedef native_duration::rep                                              native_rep;
  typedef native_duration::period                                           native_period;
  typedef std::chrono::time_point<clock_tbb_tick_count, native_duration>    native_time_point;

  static native_time_point native_now() noexcept
  {
    tbb::tick_count             time  = tbb::tick_count::now();
    tbb::tick_count::interval_t ticks = time - epoch;
    return native_time_point(native_duration( ticks ));
  }
  */

};


namespace std {
  namespace chrono {

   /// duration_values
    template <>
      struct duration_values<tbb::tick_count::interval_t>
      {
        static const tbb::tick_count::interval_t
        zero()
        { return tbb::tick_count::interval_t(); }

        // this is not well-defined, since the tick duration is only known at run-time
        // 1.e9 seconds should be a safe value
        static const tbb::tick_count::interval_t
        max()
        { return tbb::tick_count::interval_t(1.e9); }

        // this is not well-defined, since the tick duration is only known at run-time
        // -1.e9 seconds should be a safe value
        static const tbb::tick_count::interval_t
        min()
        { return tbb::tick_count::interval_t(-1.e9); }
      };

  }
}

#endif // tbb_tick_count_h
