#ifndef posix_gettimeofday_h
#define posix_gettimeofday_h

// C++ standard headers
#include <chrono>

// POSIX standard headers
#include <sys/time.h>

// based on gettimeofday(...)
struct clock_gettimeofday
{
  typedef std::chrono::microseconds                                     duration;
  typedef duration::rep                                                 rep;
  typedef duration::period                                              period;
  typedef std::chrono::time_point<clock_gettimeofday, duration>         time_point;

  static constexpr bool is_steady = false;

  static time_point now() noexcept
  {
    timeval t;
    gettimeofday(&t, nullptr);      // XXX handle possible errors

    return time_point( std::chrono::microseconds((int64_t) t.tv_sec * 1000000ll + (int64_t) t.tv_usec) );
  }

};

#endif // posix_gettimeofday_h
