#if defined __x86_64__ or defined __i386__
// 128-bit wide int is only available on x86

#ifndef posix_times_f_h
#define posix_times_f_h

// C++ standard headers
#include <chrono>

// POSIX standard headers
#include <sys/times.h>
#include <unistd.h>

// Tested on Intel Core i5 with GCC 4.7.2
//
// (time * 1000000000 / ticks_per_second) adds an overhead of almost 20 ns per call, so we pre-compute 1000000000 / ticks_per_second 
//
//    use a long long  for the ratio, shifted left by 32 bits to store extra precision;
//    then use a 128-bit wide multiplication instead of the division, and shift away the extra precision.
//    Adds a very small overhead (order of 1-2 ns) with respect to the best case above


// define macros to give hints on branch prediction
#ifndef likely
#define likely(cond) __builtin_expect(cond, 1)
#endif // likely

#ifndef unlikely
#define unlikely(cond) __builtin_expect(cond, 0)
#endif // unlikely


// based on times()
struct clock_times_cputime_f
{
  typedef std::chrono::nanoseconds                                          duration;
  typedef duration::rep                                                     rep;
  typedef duration::period                                                  period;
  typedef std::chrono::time_point<clock_times_cputime_f, duration>          time_point;

  static constexpr bool is_steady = false;

  static inline time_point now() noexcept
  {
    static const long ticks_per_second = sysconf(_SC_CLK_TCK);
    static const int64_t nanoseconds_per_tick = (1000000000ull << 32) / ticks_per_second;

    tms cputime;
    times(& cputime);
    clock_t time = cputime.tms_utime + cputime.tms_stime;
    
    __int128_t ns = ((__int128_t) time * nanoseconds_per_tick) >> 32;
    return time_point( duration( ns ));
  }

};

// based on times()
struct clock_times_realtime_f
{
  typedef std::chrono::nanoseconds                                          duration;
  typedef duration::rep                                                     rep;
  typedef duration::period                                                  period;
  typedef std::chrono::time_point<clock_times_realtime_f, duration>         time_point;

  static constexpr bool is_steady = false;

  static inline time_point now() noexcept
  {
    static const long     ticks_per_second = sysconf(_SC_CLK_TCK);
    static const int64_t nanoseconds_per_tick = (1000000000ull << 32) / ticks_per_second;

    clock_t time;
#ifdef __linux__
    time = times(nullptr);
#else
    tms cputime;
    time = times(& cputime);
#endif // __linux__
    
    __int128_t ns = ((__int128_t) time * nanoseconds_per_tick) >> 32;
    return time_point( duration( ns ));
  }

};

#endif // posix_times_f_h

#endif // defined __x86_64__ or defined __i386__
