#if defined __x86_64__ or defined __i386__
// TSC is only available on x86

// C++ standard headers
#include <chrono>
#include <cmath>

// for usleep
#include <unistd.h>

// for rdtsc, rdtscp, lfence, mfence, cpuid
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
bool has_tsc() {
  unsigned int eax, ebx, ecx, edx;
  if (__get_cpuid(0x01, & eax, & ebx, & ecx, & edx))
    return (edx & bit_TSC) != 0;
  else
    return false;
}

// check if the processor supports RDTSCP serialising instruction
bool has_rdtscp() {
  unsigned int eax, ebx, ecx, edx;
  if (__get_cpuid(0x80000001, & eax, & ebx, & ecx, & edx))
    return (edx & bit_RDTSCP) != 0;
  else
    return false;
}

// check if the processor supports the Invariant TSC feature (constant frequency TSC)
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
#ifdef __linux__
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2, 6, 26)
#define _HAS_PR_TSC_ENABLE
#endif // LINUX_VERSION_CODE >= KERNEL_VERSION(2, 6, 26)
#endif // __linux__

bool tsc_allowed() {
#if defined __linux__ and defined _HAS_PR_TSC_ENABLE
    int tsc_val = 0;
    prctl(PR_SET_TSC, PR_TSC_ENABLE);
    prctl(PR_GET_TSC, & tsc_val);
    return (tsc_val == PR_TSC_ENABLE);
#else
    return true;
#endif
}

#undef _HAS_PR_TSC_ENABLE


// calibrate TSC with respect to std::chrono::high_resolution_clock
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
    ticks[i] = rdtsc();
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

#endif // defined __x86_64__ or defined __i386__
