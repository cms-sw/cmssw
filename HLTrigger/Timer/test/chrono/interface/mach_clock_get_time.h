#ifndef mach_clock_get_time_h
#define mach_clock_get_time_h

#if defined(__APPLE__) || defined(__MACH__)

// Darwin system headers
#include <mach/mach.h>
#include <mach/clock.h>

#ifdef SYSTEM_CLOCK
#define HAVE_MACH_SYSTEM_CLOCK
#endif // SYSTEM_CLOCK

#ifdef REALTIME_CLOCK
#define HAVE_MACH_REALTIME_CLOCK
#endif // REALTIME_CLOCK

#ifdef CALENDAR_CLOCK
#define HAVE_MACH_CALENDAR_CLOCK
#endif // CALENDAR_CLOCK

// C++ standard headers
#include <chrono>

// POSIX standard headers
#include <unistd.h>
#include <time.h>


#ifdef HAVE_MACH_SYSTEM_CLOCK
// based on host_get_clock_service(SYSTEM_CLOCK) and clock_get_time(...)
struct mach_system_clock
{
  typedef std::chrono::nanoseconds                                      duration;
  typedef duration::rep                                                 rep;
  typedef duration::period                                              period;
  typedef std::chrono::time_point<mach_system_clock, duration>          time_point;

  static constexpr bool is_steady    = false;
  static constexpr bool is_available = true;

  static time_point now() noexcept
  {
    mach_timespec_t t;
    clock_get_time(clock_port, &t);

    return time_point( std::chrono::seconds(t.tv_sec) + std::chrono::nanoseconds(t.tv_nsec) );
  }

private:
  static clock_serv_t get_clock_port()
  {
    clock_serv_t clock_port;
    host_get_clock_service(mach_host_self(), SYSTEM_CLOCK, &clock_port);
    return clock_port;
  }

  static const clock_serv_t clock_port;
};
#endif // HAVE_MACH_SYSTEM_CLOCK


#ifdef HAVE_MACH_REALTIME_CLOCK
// based on host_get_clock_service(REALTIME_CLOCK) and clock_get_time(...)
struct mach_realtime_clock
{
  typedef std::chrono::nanoseconds                                      duration;
  typedef duration::rep                                                 rep;
  typedef duration::period                                              period;
  typedef std::chrono::time_point<mach_realtime_clock, duration>        time_point;

  static constexpr bool is_steady    = false;
  static constexpr bool is_available = true;

  static time_point now() noexcept
  {
    mach_timespec_t t;
    clock_get_time(clock_port, &t);

    return time_point( std::chrono::seconds(t.tv_sec) + std::chrono::nanoseconds(t.tv_nsec) );
  }

private:
  static clock_serv_t get_clock_port()
  {
    clock_serv_t clock_port;
    host_get_clock_service(mach_host_self(), REALTIME_CLOCK, &clock_port);
    return clock_port;
  }

  static const clock_serv_t clock_port;
};
#endif // HAVE_MACH_REALTIME_CLOCK


#ifdef HAVE_MACH_CALENDAR_CLOCK
// based on host_get_clock_service(CALENDAR_CLOCK) and clock_get_time(...)
struct mach_calendar_clock
{
  typedef std::chrono::nanoseconds                                      duration;
  typedef duration::rep                                                 rep;
  typedef duration::period                                              period;
  typedef std::chrono::time_point<mach_calendar_clock, duration>        time_point;

  static constexpr bool is_steady    = false;
  static constexpr bool is_available = true;

  static time_point now() noexcept
  {
    mach_timespec_t t;
    clock_get_time(clock_port, &t);

    return time_point( std::chrono::seconds(t.tv_sec) + std::chrono::nanoseconds(t.tv_nsec) );
  }

private:
  static clock_serv_t get_clock_port()
  {
    clock_serv_t clock_port;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &clock_port);
    return clock_port;
  }

  static const clock_serv_t clock_port;
};
#endif // HAVE_MACH_CALENDAR_CLOCK

#endif // defined(__APPLE__) || defined(__MACH__)

#endif // mach_clock_get_time_h
