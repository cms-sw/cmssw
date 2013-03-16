#ifndef  FWCore_Utilities_HRRealTime_H
#define  FWCore_Utilities_HRRealTime_H
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

    static __inline__ unsigned long long rdtsc(void)
    {
      unsigned long long int x;
      __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
      return x;
    }
#elif defined(__x86_64__)


    static __inline__ unsigned long long rdtsc(void)
    {
      unsigned hi, lo;
      __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
      return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
    }

#elif defined(__powerpc__)


    static __inline__ unsigned long long rdtsc(void)
    {
      unsigned long long int result=0;
      unsigned long int upper, lower,tmp;
      __asm__ volatile(
		       "0:                  \n"
		       "\tmftbu   %0           \n"
		       "\tmftb    %1           \n"
		       "\tmftbu   %2           \n"
		       "\tcmpw    %2,%0        \n"
		       "\tbne     0b         \n"
		       : "=r"(upper),"=r"(lower),"=r"(tmp)
		       );
      result = upper;
      result = result<<32;
      result = result|lower;
      
      return(result);
    }
#elif defined(__arm__)
#warning unsigned long long rdtsc(void) is not implemented on ARM architecture. Returning 0 by default.
    static __inline__ unsigned long long rdtsc(void)
    {
      return 0;
    }
#else /* defined(__arm__) */
#error The file FWCore/Utilities/interface/HRRealTime.h needs to be set up for your CPU type.
#endif
 }
}

namespace edm {

  typedef long long int HRTimeDiffType;
  typedef unsigned long long int HRTimeType;

  // High Precision real time in clock-units
  inline HRTimeType hrRealTime() {
    return details::rdtsc();
  }

}


#endif //   FWCore_Utilities__HRRealTime_H
