#if defined __x86_64__ or defined __i386__
// TSC is only available on x86

#include "interface/x86_tsc.h"
#include "interface/x86_tsc_clock.h"

const bool clock_rdtsc::is_available                = has_tsc() and tsc_allowed();
const bool clock_rdtsc::is_steady                   = has_invariant_tsc();

const bool clock_rdtsc_lfence::is_available         = has_tsc() and tsc_allowed();
const bool clock_rdtsc_lfence::is_steady            = has_invariant_tsc();

const bool clock_rdtsc_mfence::is_available         = has_tsc() and tsc_allowed();
const bool clock_rdtsc_mfence::is_steady            = has_invariant_tsc();

const bool clock_rdtscp::is_available               = has_rdtscp() and tsc_allowed();
const bool clock_rdtscp::is_steady                  = has_invariant_tsc();

const bool clock_serialising_rdtsc::is_available    = has_tsc() and tsc_allowed();
const bool clock_serialising_rdtsc::is_steady       = has_invariant_tsc();

#endif // defined __x86_64__ or defined __i386__
