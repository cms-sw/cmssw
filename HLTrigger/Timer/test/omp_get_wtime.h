#ifndef omp_get_wtime_h
#define omp_get_wtime_h

// C++ standard headers
#include <chrono>

// OpenMP headers
#include <omp.h>

// OpenMP based clock
struct clock_omp_get_wtime
{
  // std::chrono interface
  typedef std::chrono::duration<double>                                     duration;
  typedef duration::rep                                                     rep;
  typedef duration::period                                                  period;
  typedef std::chrono::time_point<clock_omp_get_wtime, duration>            time_point;

  static constexpr bool is_steady = false;

  static time_point now() noexcept
  {
    double seconds = omp_get_wtime();
    return time_point(duration( seconds ));
  }
};

#endif // omp_get_wtime_h
