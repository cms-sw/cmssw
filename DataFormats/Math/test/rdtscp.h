#ifndef RDPSCP_H
#define RDPSCP_H
// performance test
#ifndef __arm__
#include <x86intrin.h>
#include <cpuid.h>
#ifdef __clang__
bool has_rdtscp() { return true;}
/** CPU cycles since processor startup */
inline uint64_t rdtsc() {
uint32_t lo, hi;
/* We cannot use "=A", since this would use %rax on x86_64 */
__asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
return (uint64_t)hi << 32 | lo;
}
#else
// CPUID, EAX = 0x80000001, EDX values
#ifndef bit_RDTSCP
#define bit_RDTSCP          (1 << 27)
#endif
namespace {
  inline
  bool has_rdtscp() {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(0x80000001, & eax, & ebx, & ecx, & edx))
      return (edx & bit_RDTSCP) != 0;
    else
      return false;
  }
  unsigned int rdtscp_val=0;
  inline volatile unsigned long long rdtsc() {
    return __rdtscp(&rdtscp_val);
  }
}
#endif
#else  // arm
namespace {
inline bool has_rdtscp() { return false;}
inline volatile unsigned long long rdtsc() {return 0;}
}
#endif // arm

#endif
