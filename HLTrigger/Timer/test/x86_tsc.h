#if defined __x86_64__ or defined __i386__
// TSC is only available on x86

#ifndef x86_tsc_h
#define x86_tsc_h

#include <cstdint>

// GCC and ICC provide intrinsics for rdtsc and rdtscp
#include <x86intrin.h>

extern inline uint64_t rdtsc(void)
{
    return __rdtsc();
}

extern inline uint64_t rdtscp(uint32_t *aux)
{
    return __rdtscp(aux);
}

bool has_tsc();
bool has_rdtscp();
bool has_invariant_tsc();
bool tsc_allowed();

double calibrate_tsc_hz();

#endif // x86_tsc_h

#endif // defined __x86_64__ or defined __i386__
