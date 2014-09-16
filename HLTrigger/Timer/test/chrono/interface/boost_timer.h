#ifndef boost_timer_h
#define boost_timer_h

// C++ headers
#include <chrono>

// BOOST headers
#include <boost/timer/timer.hpp>

// based on boost::timer::cpu_timer
struct clock_boost_timer_cputime
{
  typedef std::chrono::nanoseconds                                      duration;
  typedef duration::rep                                                 rep;
  typedef duration::period                                              period;
  typedef std::chrono::time_point<clock_boost_timer_cputime, duration>  time_point;

  static constexpr bool is_steady = false;

  static time_point now() noexcept
  {
    static boost::timer::cpu_timer timer;
    auto const & elapsed = timer.elapsed();

    return time_point( std::chrono::nanoseconds(elapsed.user + elapsed.system) );
  }

};


// based on boost::timer::cpu_timer
struct clock_boost_timer_realtime
{
  typedef std::chrono::nanoseconds                                      duration;
  typedef duration::rep                                                 rep;
  typedef duration::period                                              period;
  typedef std::chrono::time_point<clock_boost_timer_realtime, duration> time_point;

  static constexpr bool is_steady = false;

  static time_point now() noexcept
  {
    static boost::timer::cpu_timer timer;
    auto const & elapsed = timer.elapsed();

    return time_point( std::chrono::nanoseconds(elapsed.wall) );
  }

};

#endif // boost_timer_h
