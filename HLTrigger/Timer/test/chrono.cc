// macro to check for GCC version
#define GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)

// C++ headers
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <chrono>

// for uname
#include <sys/utsname.h>

#ifdef __linux__
// for gnu_get_libc_version
#include <gnu/libc-version.h>
#endif // __linux__

// other clocks
#include "posix_clock.h"
#include "posix_clock_gettime.h"
#include "posix_gettimeofday.h"
#include "posix_times.h"
#include "posix_times_f.h"
#include "posix_times_d.h"
#include "posix_getrusage.h"
#include "mach_clock_get_time.h"
#include "mach_absolute_time.h"
#include "x86_tsc_clock.h"
#include "boost_timer.h"
#include "tbb_tick_count.h"
#include "omp_get_wtime.h"

#include "benchmark.h"


void init_timers(std::vector<BenchmarkBase *> & timers) 
{
  // std::chrono timers
#if GCC_VERSION >= 40700
  // C++11 clock name
  timers.push_back(new Benchmark<std::chrono::steady_clock>("std::chrono::steady_clock"));
#else
  // pre-C++11 clock name
  timers.push_back(new Benchmark<std::chrono::monotonic_clock>("std::chrono::monotonic_clock"));
#endif
  timers.push_back(new Benchmark<std::chrono::system_clock>("std::chrono::system_clock"));
  timers.push_back(new Benchmark<std::chrono::high_resolution_clock>("std::chrono::high_resolution_clock"));

  // POSIX clock_gettime
#ifdef HAVE_POSIX_CLOCK_REALTIME
  if (clock_gettime_realtime::is_available)
    timers.push_back(new Benchmark<clock_gettime_realtime>("clock_gettime(CLOCK_REALTIME)"));
#endif // HAVE_POSIX_CLOCK_REALTIME
#ifdef HAVE_POSIX_CLOCK_MONOTONIC
  if (clock_gettime_monotonic::is_available)
    timers.push_back(new Benchmark<clock_gettime_monotonic>("clock_gettime(CLOCK_MONOTONIC)"));
#endif // HAVE_POSIX_CLOCK_MONOTONIC
#ifdef HAVE_POSIX_CLOCK_MONOTONIC_RAW
  if (clock_gettime_monotonic_raw::is_available)
    timers.push_back(new Benchmark<clock_gettime_monotonic_raw>("clock_gettime(CLOCK_MONOTONIC_RAW)"));
#endif // HAVE_POSIX_CLOCK_MONOTONIC_RAW
#ifdef HAVE_POSIX_CLOCK_PROCESS_CPUTIME_ID
  if (clock_gettime_process_cputime::is_available)
    timers.push_back(new Benchmark<clock_gettime_process_cputime>("clock_gettime(CLOCK_PROCESS_CPUTIME_ID)"));
#endif // HAVE_POSIX_CLOCK_PROCESS_CPUTIME_ID
#ifdef HAVE_POSIX_CLOCK_THREAD_CPUTIME_ID
  if (clock_gettime_thread_cputime::is_available)
    timers.push_back(new Benchmark<clock_gettime_thread_cputime>("clock_gettime(CLOCK_THREAD_CPUTIME_ID)"));
#endif // HAVE_POSIX_CLOCK_THREAD_CPUTIME_ID

  // POSIX gettimeofday
  timers.push_back(new Benchmark<clock_gettimeofday>("gettimeofday()"));

  // POSIX times
  timers.push_back(new Benchmark<clock_times_realtime>("times() (wall-clock time)"));
  timers.push_back(new Benchmark<clock_times_cputime>("times() (cpu time)"));
  timers.push_back(new Benchmark<clock_times_realtime_d>("times() (wall-clock time) (using double)"));
  timers.push_back(new Benchmark<clock_times_cputime_d>("times() (cpu time) (using double)"));
#if defined __x86_64__ or defined __i386__
// 128-bit wide int is only available on x86
  timers.push_back(new Benchmark<clock_times_realtime_f>("times() (wall-clock time) (using fixed math)"));
  timers.push_back(new Benchmark<clock_times_cputime_f>("times() (cpu time) (using fixed math)"));
#endif // defined __x86_64__ or defined __i386__

  // POSIX clock
  timers.push_back(new Benchmark<clock_clock>("clock()"));

  // POSIX getrusage
  timers.push_back(new Benchmark<clock_getrusage_self>("getrusage(RUSAGE_SELF)"));
#ifdef HAVE_POSIX_CLOCK_GETRUSAGE_THREAD
  timers.push_back(new Benchmark<clock_getrusage_thread>("getrusage(RUSAGE_THREAD)"));
#endif // HAVE_POSIX_CLOCK_GETRUSAGE_THREAD

  // MACH clock_get_time
#ifdef HAVE_MACH_SYSTEM_CLOCK
  if (mach_system_clock::is_available)
    timers.push_back(new Benchmark<mach_system_clock>("host_get_clock_service(SYSTEM_CLOCK), clock_get_time(...)"));
#endif // HAVE_MACH_SYSTEM_CLOCK
#ifdef HAVE_MACH_REALTIME_CLOCK
  if (mach_realtime_clock::is_available)
    timers.push_back(new Benchmark<mach_realtime_clock>("host_get_clock_service(REALTIME_CLOCK), clock_get_time(...)"));
#endif // HAVE_MACH_REALTIME_CLOCK
#ifdef HAVE_MACH_CALENDAR_CLOCK
  if (mach_calendar_clock::is_available)
    timers.push_back(new Benchmark<mach_calendar_clock>("host_get_clock_service(CALENDAR_CLOCK), clock_get_time(...)"));
#endif // HAVE_MACH_CALENDAR_CLOCK
#ifdef HAVE_MACH_ABSOLUTE_TIME
  if (mach_absolute_time_clock::is_available) {
    timers.push_back(new Benchmark<mach_absolute_time_clock>("mach_absolute_time() (using nanoseconds)"));
    timers.push_back(new Benchmark<mach_absolute_time_clock_native>("mach_absolute_time() (native)"));
  }
#endif // HAVE_MACH_ABSOLUTE_TIME

#if defined __x86_64__ or defined __i386__
// TSC is only available on x86
  
  // read TSC clock frequency
  std::stringstream buffer;
  buffer << std::fixed << std::setprecision(3) << (tsc_tick::ticks_per_second / 1.e6) << " MHz";
  std::string tsc_freq = buffer.str();

  // x86 DST-based clock (via std::chrono::nanosecond)
  if (clock_rdtsc::is_available)
    timers.push_back(new Benchmark<clock_rdtsc>("RDTSC (" + tsc_freq + ") (using nanoseconds)"));
  if (clock_rdtsc_lfence::is_available)
    timers.push_back(new Benchmark<clock_rdtsc_lfence>("LFENCE; RDTSC (" + tsc_freq + ") (using nanoseconds)"));
  if (clock_rdtsc_mfence::is_available)
    timers.push_back(new Benchmark<clock_rdtsc_mfence>("MFENCE; RDTSC (" + tsc_freq + ") (using nanoseconds)"));
  if (clock_rdtscp::is_available)
    timers.push_back(new Benchmark<clock_rdtscp>("RDTSCP (" + tsc_freq + ") (using nanoseconds)"));
  // x86 DST-based clock (native)
  if (clock_rdtsc::is_available)
    timers.push_back(new Benchmark<clock_rdtsc_native>("RDTSC (" + tsc_freq + ") (native)"));
  if (clock_rdtsc_lfence::is_available)
    timers.push_back(new Benchmark<clock_rdtsc_lfence_native>("LFENCE; RDTSC (" + tsc_freq + ") (native)"));
  if (clock_rdtsc_mfence::is_available)
    timers.push_back(new Benchmark<clock_rdtsc_mfence_native>("MFENCE; RDTSC (" + tsc_freq + ") (native)"));
  if (clock_rdtscp::is_available)
    timers.push_back(new Benchmark<clock_rdtscp_native>("RDTSCP (" + tsc_freq + ") (native)"));

#endif // defined __x86_64__ or defined __i386__

  // boost timer clock
  timers.push_back(new Benchmark<clock_boost_timer_realtime>("boost::timer (wall-clock time)"));
  timers.push_back(new Benchmark<clock_boost_timer_cputime>("boost::timer (cpu time)"));

  // TBB tick_count (this interface does not expose the underlying type, so it cannot easily be used to build a "native" clock interface)
  timers.push_back(new Benchmark<clock_tbb_tick_count>("tbb::tick_count"));

  // OpenMP timer
  timers.push_back(new Benchmark<clock_omp_get_wtime>("omp_get_wtime"));
}


std::string read_kernel_version() {
  struct utsname names;
  if (not uname(& names)) {
    std::stringstream buffer;
    buffer << names.sysname << " " << names.release << " " << names.machine;
    return buffer.str();
  } else {
    return std::string("unknown");
  }
}


#ifdef __linux__
std::string read_glibc_version() {
  std::string version( gnu_get_libc_version() );
  return version;
}


std::string read_clock_source() {
  std::ifstream current_clocksource("/sys/devices/system/clocksource/clocksource0/current_clocksource");
  if (current_clocksource.good()) {
    std::string value;
    current_clocksource >> value;
    return value;
  } else {
    return std::string("unknown");
  }
}
#endif // __linux__


int main(void) {
  std::vector<BenchmarkBase *> timers;
  init_timers(timers);

  std::cout << read_kernel_version() << std::endl;
#ifdef __linux__
  std::cout << "glibc version: " << read_glibc_version() << std::endl;
  std::cout << "clock source: " << read_clock_source() << std::endl;
#endif // __linux__

  std::cout << "For each timer the resolution reported is the MINIMUM (MEDIAN) (MEAN +/- its STDDEV) of the increments measured during the test." << std::endl << std::endl; 

  for (BenchmarkBase * timer: timers) {
    timer->measure();
    timer->compute();
    timer->report();
  }

  return 0;
}
