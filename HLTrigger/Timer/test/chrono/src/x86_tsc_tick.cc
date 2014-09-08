#if defined __x86_64__ or defined __i386__
// TSC is only available on x86

#include "interface/x86_tsc.h"
#include "interface/x86_tsc_tick.h"

const double  tsc_tick::ticks_per_second = calibrate_tsc_hz();
const double  tsc_tick::seconds_per_tick = 1. / tsc_tick::ticks_per_second;
const int64_t tsc_tick::nanoseconds_per_tick_shifted = (1000000000ll << 32) / tsc_tick::ticks_per_second;
//const int64_t tsc_tick::ticks_per_nanosecond_shifted = (int64_t) ((((__int128_t) tsc_tick::ticks_per_second) << 32) / 1000000000ll);
const int64_t tsc_tick::ticks_per_nanosecond_shifted = (int64_t) llrint(tsc_tick::ticks_per_second * 4.294967296);

#endif // defined __x86_64__ or defined __i386__
