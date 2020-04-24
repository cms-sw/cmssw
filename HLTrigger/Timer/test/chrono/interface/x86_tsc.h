#if defined __x86_64__ or defined __i386__
// TSC is only available on x86

#ifndef x86_tsc_h
#define x86_tsc_h

#include <cstdint>

#if defined __clang__
// clang does not define the __rdtsc and __rdtscp intrinsic, although it does
// define __builtin_readcyclecounter() which is a likely replacement for __rdtsc()

extern inline uint64_t rdtsc(void)
{
    uint32_t eax, edx;
    asm("rdtsc" : "=a" (eax), "=d" (edx));
    return ((uint64_t) edx << 32) | (uint64_t) eax;
}

extern inline uint64_t rdtscp(uint32_t *aux)
{
    uint32_t eax, edx;
    uint64_t rcx;
    asm("rdtscp" : "=a" (eax), "=d" (edx), "=c" (rcx));
    *aux = rcx;
    return ((uint64_t) edx << 32) | (uint64_t) eax;
}

#elif defined __GNUC__
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

#else
#  error "Unsupported compiler"
#endif // __clang__ / __GNUC__

bool has_tsc();
bool has_rdtscp();
bool has_invariant_tsc();
bool tsc_allowed();

double calibrate_tsc_hz();


// IFUNC support requires GCC >= 4.6.0 and GLIBC >= 2.11.1
#if ( defined __GNUC__ && (__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6) ) \
  and ( defined __GLIBC__ && (__GLIBC__ > 2) || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 11) )

// processor specific serialising access to the TSC
extern uint64_t serialising_rdtsc(void);

#else

// processor specific serialising access to the TSC
extern uint64_t (*serialising_rdtsc)(void);

#endif // IFUNC support

#endif // x86_tsc_h

#endif // defined __x86_64__ or defined __i386__
