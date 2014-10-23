#if defined __x86_64__ or defined __i386__
// TSC is only available on x86

#ifndef x86_tsc_clock_h
#define x86_tsc_clock_h

// C++ standard headers
#include <chrono>

// for rdtscp, rdtscp, lfence, mfence
#include <x86intrin.h>

// for tsc_tick, etc.
#include "interface/x86_tsc.h"
#include "interface/x86_tsc_tick.h"

// TSC-based clock, using rdtsc (non-serialising)
struct clock_rdtsc
{
  // std::chrono interface
  typedef std::chrono::nanoseconds                                      duration;
  typedef duration::rep                                                 rep;
  typedef duration::period                                              period;
  typedef std::chrono::time_point<clock_rdtsc, duration>                time_point;

  static const bool is_steady;
  static const bool is_available;

  static time_point now() noexcept
  {
    int64_t    ticks = rdtsc();
    rep        ns    = tsc_tick::to_nanoseconds(ticks);
    time_point time  = time_point(duration(ns));
    return time;
  }
};


// TSC-based clock, using lfence as serialising instruction
struct clock_rdtsc_lfence
{
  // std::chrono interface
  typedef std::chrono::nanoseconds                                      duration;
  typedef duration::rep                                                 rep;
  typedef duration::period                                              period;
  typedef std::chrono::time_point<clock_rdtsc_lfence, duration>         time_point;

  static const bool is_steady;
  static const bool is_available;

  static time_point now() noexcept
  {
    _mm_lfence();
    int64_t    ticks = rdtsc();
    rep        ns    = tsc_tick::to_nanoseconds(ticks);
    time_point time  = time_point(duration(ns));
    return time;
  }
};

// TSC-based clock, using mfence as serialising instruction
struct clock_rdtsc_mfence
{
  // std::chrono interface
  typedef std::chrono::nanoseconds                                      duration;
  typedef duration::rep                                                 rep;
  typedef duration::period                                              period;
  typedef std::chrono::time_point<clock_rdtsc_mfence, duration>         time_point;

  static const bool is_steady;
  static const bool is_available;

  static time_point now() noexcept
  {
    _mm_mfence();
    int64_t    ticks = rdtsc();
    rep        ns    = tsc_tick::to_nanoseconds(ticks);
    time_point time  = time_point(duration(ns));
    return time;
  }
};


// TSC-based clock, using rdtscp as serialising instruction
struct clock_rdtscp
{
  // std::chrono interface
  typedef std::chrono::nanoseconds                                      duration;
  typedef duration::rep                                                 rep;
  typedef duration::period                                              period;
  typedef std::chrono::time_point<clock_rdtscp, duration>               time_point;

  static const bool is_steady;
  static const bool is_available;

  static time_point now() noexcept
  {
    unsigned int id;
    int64_t    ticks = rdtscp(& id);
    rep        ns    = tsc_tick::to_nanoseconds(ticks);
    time_point time  = time_point(duration(ns));
    return time;
  }
};


// TSC-based clock, determining at run-time the best strategy to serialise the reads from the TSC
struct clock_serialising_rdtsc
{
  // std::chrono interface
  typedef std::chrono::nanoseconds                                      duration;
  typedef duration::rep                                                 rep;
  typedef duration::period                                              period;
  typedef std::chrono::time_point<clock_serialising_rdtsc, duration>    time_point;

  static const bool is_steady;
  static const bool is_available;

  static time_point now() noexcept
  {
    int64_t    ticks = serialising_rdtsc();
    rep        ns    = tsc_tick::to_nanoseconds(ticks);
    time_point time  = time_point(duration(ns));
    return time;
  }
};


#endif // x86_tsc_clock_h

#endif // defined __x86_64__ or defined __i386__
