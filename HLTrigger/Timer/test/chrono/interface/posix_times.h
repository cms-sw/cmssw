#ifndef posix_times_h
#define posix_times_h

// C++ standard headers
#include <chrono>

// POSIX standard headers
#include <sys/times.h>
#include <unistd.h>

// Tested on Intel Core i5 with GCC 4.7.2
//
// (time * 1000000000 / ticks_per_second) adds an overhead of almost 20 ns per call, so we pre-compute 1000000000 / ticks_per_second 
//
//    use a long for the ratio, check if it is accurate or not, and only if it is accurate use it instead of doing the division.
//    When accurate, (time * nanoseconds_per_tick) can be done basically for free


// define macros to give hints on branch prediction
#ifndef likely
#define likely(cond) __builtin_expect(cond, 1)
#endif // likely

#ifndef unlikely
#define unlikely(cond) __builtin_expect(cond, 0)
#endif // unlikely


// based on times()
struct clock_times_cputime
{
  typedef std::chrono::nanoseconds                                          duration;
  typedef duration::rep                                                     rep;
  typedef duration::period                                                  period;
  typedef std::chrono::time_point<clock_times_cputime, duration>            time_point;

  static constexpr bool is_steady = false;

  static inline time_point now() noexcept
  {
    static const long ticks_per_second = sysconf(_SC_CLK_TCK);
    static const long nanoseconds_per_tick = 1000000000l / ticks_per_second;
    static const bool accurate = ((1000000000l % ticks_per_second) == 0);

    tms cputime;
    times(& cputime);
    clock_t time = cputime.tms_utime + cputime.tms_stime;
    
    int64_t ns = likely(accurate) ? time * nanoseconds_per_tick : time * 1000000000l / ticks_per_second;
    return time_point( duration( ns ));
  }

};

// based on times()
struct clock_times_realtime
{
  typedef std::chrono::nanoseconds                                          duration;
  typedef duration::rep                                                     rep;
  typedef duration::period                                                  period;
  typedef std::chrono::time_point<clock_times_realtime, duration>           time_point;

  static constexpr bool is_steady = false;

  static inline time_point now() noexcept
  {
    static const long ticks_per_second = sysconf(_SC_CLK_TCK);
    static const long nanoseconds_per_tick = 1000000000l / ticks_per_second;
    static const bool accurate = ((1000000000l % ticks_per_second) == 0);

    clock_t time;
#ifdef __linux__
    time = times(nullptr);
#else
    tms cputime;
    time = times(& cputime);
#endif // __linux__
    
    int64_t ns = likely(accurate) ? time * nanoseconds_per_tick : time * 1000000000l / ticks_per_second;
    return time_point( duration( ns ));
  }

};

#endif // posix_times_h
