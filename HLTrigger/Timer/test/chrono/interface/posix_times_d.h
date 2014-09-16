#ifndef posix_times_d_h
#define posix_times_d_h

// C++ standard headers
#include <chrono>

// POSIX standard headers
#include <sys/times.h>
#include <unistd.h>

// based on times()
struct clock_times_cputime_d
{
  typedef std::chrono::duration<double>                                     duration;
  typedef duration::rep                                                     rep;
  typedef duration::period                                                  period;
  typedef std::chrono::time_point<clock_times_cputime_d, duration>          time_point;

  static constexpr bool is_steady = false;

  static inline time_point now() noexcept
  {
    static const double ticks_per_second = sysconf(_SC_CLK_TCK);
    static const double seconds_per_tick = 1. / ticks_per_second;

    tms cputime;
    times(& cputime);
    clock_t time = cputime.tms_utime + cputime.tms_stime;
    
    return time_point( duration( (double) time * seconds_per_tick ) );
  }

};

// based on times()
struct clock_times_realtime_d
{
  typedef std::chrono::duration<double>                                     duration;
  typedef duration::rep                                                     rep;
  typedef duration::period                                                  period;
  typedef std::chrono::time_point<clock_times_realtime_d, duration>         time_point;

  static constexpr bool is_steady = false;

  static inline time_point now() noexcept
  {
    static const double ticks_per_second = sysconf(_SC_CLK_TCK);
    static const double seconds_per_tick = 1. / ticks_per_second;

    clock_t time;
#ifdef __linux__
    time = times(nullptr);
#else
    tms cputime;
    time = times(& cputime);
#endif // __linux__
    
    return time_point( duration( (double) time * seconds_per_tick ) ); 
  }

};


#endif // posix_times_d_h
