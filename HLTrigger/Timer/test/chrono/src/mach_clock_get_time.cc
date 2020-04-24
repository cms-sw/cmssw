#if defined(__APPLE__) || defined(__MACH__)

// Darwin system headers
#include <mach/mach.h>
#include <mach/clock.h>

#include "interface/mach_clock_get_time.h"

#ifdef HAVE_MACH_SYSTEM_CLOCK
// based on host_get_clock_service(SYSTEM_CLOCK) and clock_get_time(...)
const clock_serv_t mach_system_clock::clock_port = mach_system_clock::get_clock_port();
#endif // HAVE_MACH_SYSTEM_CLOCK

#ifdef HAVE_MACH_REALTIME_CLOCK
// based on host_get_clock_service(REALTIME_CLOCK) and clock_get_time(...)
const clock_serv_t mach_realtime_clock::clock_port = mach_realtime_clock::get_clock_port();
#endif // HAVE_MACH_REALTIME_CLOCK


#ifdef HAVE_MACH_CALENDAR_CLOCK
// based on host_get_clock_service(CALENDAR_CLOCK) and clock_get_time(...)
const clock_serv_t mach_calendar_clock::clock_port = mach_calendar_clock::get_clock_port();
#endif // HAVE_MACH_CALENDAR_CLOCK

#endif // defined(__APPLE__) || defined(__MACH__)
