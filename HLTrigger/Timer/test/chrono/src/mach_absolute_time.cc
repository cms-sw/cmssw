#if defined(__APPLE__) || defined(__MACH__)

// C++ standard headers
#include <chrono>
#include <cmath>

// Darwin system headers
#include <mach/mach.h>
#include <mach/mach_time.h>

#include "interface/mach_absolute_time.h"

// read the calibration of mach_absolute_time
static
double calibrate_ticks_per_second() {
  mach_timebase_info_data_t timebase_info;
  mach_timebase_info(& timebase_info);
  return 1.e9 * timebase_info.denom / timebase_info.numer;
}

static
int64_t calibrate_nanoseconds_per_tick_shifted() {
  mach_timebase_info_data_t timebase_info;
  mach_timebase_info(& timebase_info);
  return (1ll << 32) * timebase_info.numer / timebase_info.denom;
}

static
int64_t calibrate_ticks_per_nanosecond_shifted() {
  mach_timebase_info_data_t timebase_info;
  mach_timebase_info(& timebase_info);
  return (1ll << 32) * timebase_info.denom / timebase_info.numer;
}

const double  mach_absolute_time_tick::ticks_per_second = calibrate_ticks_per_second();
const double  mach_absolute_time_tick::seconds_per_tick = 1. / mach_absolute_time_tick::ticks_per_second;
const int64_t mach_absolute_time_tick::nanoseconds_per_tick_shifted = calibrate_nanoseconds_per_tick_shifted();
const int64_t mach_absolute_time_tick::ticks_per_nanosecond_shifted = calibrate_ticks_per_nanosecond_shifted();


#endif // defined(__APPLE__) || defined(__MACH__)
