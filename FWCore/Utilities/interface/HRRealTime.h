#ifndef FWCore_Utilities_HRRealTime_H
#define FWCore_Utilities_HRRealTime_H
/*  
 *  High Resolution Real Timer
 *  inline high-resolution real timer
 *  to be used for precise measurements of performance of
 *  "small" chunks of code.  
 *
 *  returns time in "nominal" cpu-clock unit
 *  on most recent hw-architecure it is compensated for clock-rate variations
 *  so to get seconds it shall be multiplied for a nominal cpu-clock unit
 *  Performance comparison make sense only if the clock-rate has been fixed
 */

namespace edm {
  namespace details {

    //
    //  defines "rdtsc"
    //
#if defined(__i386__)

    static __inline__ unsigned long long rdtsc(void) {
      unsigned long long int x;
      __asm__ volatile(".byte 0x0f, 0x31" : "=A"(x));
      return x;
    }
#elif defined(__x86_64__)

    static __inline__ unsigned long long rdtsc(void) {
      unsigned hi, lo;
      __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
      return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
    }

#elif defined(__powerpc__)

    static __inline__ unsigned long long rdtsc(void) {
      unsigned long long int result = 0;
      unsigned long int upper, lower, tmp;
      __asm__ volatile(
          "0:                  \n"
          "\tmftbu   %0           \n"
          "\tmftb    %1           \n"
          "\tmftbu   %2           \n"
          "\tcmpw    %2,%0        \n"
          "\tbne     0b         \n"
          : "=r"(upper), "=r"(lower), "=r"(tmp));
      result = upper;
      result = result << 32;
      result = result | lower;

      return (result);
    }
#elif defined(__arm__)
#warning unsigned long long rdtsc(void) is not implemented on ARMv7 architecture. Returning 0 by default.
    static __inline__ unsigned long long rdtsc(void) { return 0; }
#elif defined(__aarch64__)
    static __inline__ unsigned long long rdtsc(void) {
      // We will be reading CNTVCT_EL0 (the virtual counter), which is prepared for us by OS.
      // The system counter sits outside multiprocessor in SOC and runs on a different frequency.
      // Increments at a fixed frequency, typically in the range 1-50MHz.
      // Applications can figure out system counter configuration via CNTFRQ_EL0.
      //
      // Notice:
      // Reads of CNTVCT_EL0 can occur speculatively and out of order relative to other
      // instructions executed on the same PE.
      // For example, if a read from memory is used to obtain a signal from another agent
      // that indicates that CNTVCT_EL0 must be read, an ISB is used to ensure that the
      // read of CNTVCT_EL0 occurs after the signal has been read from memory
      //
      // More details:
      // Chapter D6: The Generic Timer in AArch64 state
      // ARM DDI 0487B.a, ID033117 (file: DDI0487B_a_armv8_arm.pdf)
      unsigned long long ret;  // unsigned 64-bit value
      __asm__ __volatile__("isb; mrs %0, cntvct_el0" : "=r"(ret));
      return ret;
    }
#else
#error The file FWCore/Utilities/interface/HRRealTime.h needs to be set up for your CPU type.
#endif
  }  // namespace details
}  // namespace edm

namespace edm {

  typedef long long int HRTimeDiffType;
  typedef unsigned long long int HRTimeType;

  // High Precision real time in clock-units
  inline HRTimeType hrRealTime() { return details::rdtsc(); }

}  // namespace edm

#endif  //   FWCore_Utilities__HRRealTime_H
