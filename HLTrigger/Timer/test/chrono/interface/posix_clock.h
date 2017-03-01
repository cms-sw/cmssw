#ifndef posix_clock_h
#define posix_clock_h

// C++ standard headers
#include <chrono>

// POSIX standard headers
#include <time.h>

// based on clock()
struct clock_clock
{
  typedef std::chrono::duration<long long, std::ratio<1, CLOCKS_PER_SEC>>   duration;
  typedef duration::rep                                                     rep;
  typedef duration::period                                                  period;
  typedef std::chrono::time_point<clock_clock, duration>                    time_point;

  static constexpr bool is_steady = false;

  static inline time_point now() noexcept
  {
    clock_t time = clock();
    return time_point( duration( time ));
  }

};

#endif // posix_clock_h
