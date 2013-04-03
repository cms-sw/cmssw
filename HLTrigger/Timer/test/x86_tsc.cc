#ifdef __x86_64__
// TSC is only available on x86

// C++ standard headers
#include <chrono>
#include <cmath>

// for usleep
#include <unistd.h>

// for rdtscp, rdtscp, lfence, mfence, cpuid
#include <x86intrin.h>
#include <cpuid.h>

#ifdef __linux__
#include <linux/version.h>
#include <sys/prctl.h>
#endif // __linux__

#include "x86_tsc.h"


// CPUID, EAX = 0x01, EDX values
#ifndef bit_TSC
#define bit_TSC             (1 << 4)
#endif

// CPUID, EAX = 0x80000001, EDX values
#ifndef bit_RDTSCP
#define bit_RDTSCP          (1 << 27)
#endif

// CPUID, EAX = 0x80000007, EDX values
#ifndef bit_InvariantTSC
#define bit_InvariantTSC    (1 << 8)
#endif


// check if the processor has a TSC (Time Stamp Counter) and supports the RDTSC instruction
static
bool has_tsc() {
  unsigned int eax, ebx, ecx, edx;
  if (__get_cpuid(0x01, & eax, & ebx, & ecx, & edx))
    return (edx & bit_TSC) != 0;
  else
    return false;
}

// check if the processor supports RDTSCP serialising instruction
static
bool has_rdtscp() {
  unsigned int eax, ebx, ecx, edx;
  if (__get_cpuid(0x80000001, & eax, & ebx, & ecx, & edx))
    return (edx & bit_RDTSCP) != 0;
  else
    return false;
}

// check if the processor supports the Invariant TSC feature (constant frequency TSC)
static
bool has_invariant_tsc() {
  unsigned int eax, ebx, ecx, edx;
  if (__get_cpuid(0x80000007, & eax, & ebx, & ecx, & edx))
    return (edx & bit_InvariantTSC) != 0;
  else
    return false;
}


// Check if the RDTSC and RDTSCP instructions are allowed in user space.
// This is controlled by the x86 control register 4, bit 4 (CR4.TSD), but that is only readable by the kernel.
// On Linux, the flag can be read (and possibly set) via the prctl interface.
static
bool tsc_allowed() {
#ifdef __linux__
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2, 6, 26)
    int tsc_val = 0;
    prctl(PR_SET_TSC, PR_TSC_ENABLE);
    prctl(PR_GET_TSC, & tsc_val);
    return (tsc_val == PR_TSC_ENABLE);
#else  // LINUX_VERSION_CODE >= KERNEL_VERSION(2, 6, 26)
    return true;
#endif // LINUX_VERSION_CODE >= KERNEL_VERSION(2, 6, 26)
#else  // __linux__
    return true;
#endif // __linux__
}


// calibrate TSC with respect to std::chrono::high_resolution_clock
static
double calibrate_tsc_hz() {
  if (not has_tsc() or not tsc_allowed())
    return 0;

  constexpr unsigned int sample_size = 1000;        // 1000 samples
  constexpr unsigned int sleep_time  = 1000;        //    1 ms
  unsigned long long ticks[sample_size];
  double             times[sample_size];

  auto reference = std::chrono::high_resolution_clock::now();
  for (unsigned int i = 0; i < sample_size; ++i) {
    usleep(sleep_time);
    ticks[i] = __rdtsc();
    times[i] = std::chrono::duration_cast<std::chrono::duration<double>>( std::chrono::high_resolution_clock::now() - reference ).count();
  }

  double mean_x = 0, mean_y = 0;
  for (unsigned int i = 0; i < sample_size; ++i) {
    mean_x += (double) times[i];
    mean_y += (double) ticks[i];
  }
  mean_x /= (double) sample_size;
  mean_y /= (double) sample_size;

  double sigma_xy = 0, sigma_xx = 0;
  for (unsigned int i = 0; i < sample_size; ++i) {
    sigma_xx += (double) (times[i] - mean_x) * (double) (times[i] - mean_x);
    sigma_xy += (double) (times[i] - mean_x) * (double) (ticks[i] - mean_y);
  }

  // ticks per second
  return sigma_xy / sigma_xx;
}


const double  tsc_tick::ticks_per_second = calibrate_tsc_hz();
const double  tsc_tick::seconds_per_tick = 1. / tsc_tick::ticks_per_second;
const int64_t tsc_tick::nanoseconds_per_tick_shifted = (1000000000ll << 32) / tsc_tick::ticks_per_second;
const int64_t tsc_tick::ticks_per_nanosecond_shifted = (int64_t) ((((__int128_t) tsc_tick::ticks_per_second) << 32) / 1000000000ll);


const bool clock_rdtsc::is_available        = has_tsc() and tsc_allowed();
const bool clock_rdtsc::is_steady           = has_invariant_tsc();

const bool clock_rdtsc_lfence::is_available = has_tsc() and tsc_allowed();
const bool clock_rdtsc_lfence::is_steady    = has_invariant_tsc();

const bool clock_rdtsc_mfence::is_available = has_tsc() and tsc_allowed();
const bool clock_rdtsc_mfence::is_steady    = has_invariant_tsc();

const bool clock_rdtscp::is_available       = has_rdtscp() and tsc_allowed();
const bool clock_rdtscp::is_steady          = has_invariant_tsc();

#endif // __x86_64__
