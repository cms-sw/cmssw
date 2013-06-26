#ifndef FWCore_Utilities_math_h
#define FWCore_Utilities_math_h

#include <cmath>
#include <sys/types.h>
#include "math_private.h"

namespace edm 
{

  namespace detail
  {
    inline bool isnan(float x)
    {
      u_int32_t wx;
      
      GET_FLOAT_WORD (wx, x);
      wx &= 0x7fffffff;
      return (bool)(wx > 0x7f800000);
    }
    
    inline bool isnan(double x)
    {
      u_int32_t hx, lx;
      
      EXTRACT_WORDS (hx, lx, x);
      lx |= hx & 0xfffff;
      hx &= 0x7ff00000;
      return (bool)(hx == 0x7ff00000) && (lx != 0);
    }
    
    inline bool isnan(long double x)
    {
      u_int32_t ex, hx, lx;
      
      GET_LDOUBLE_WORDS (ex, hx, lx, x);
      ex &= 0x7fff;
      return (bool)((ex == 0x7fff) && ((hx & 0x7fffffff) | lx));
    }
  }


  template <class FP> inline bool asm_isnan(FP x)
  {
    // I do not know of a preprocessor symbol used to identify the
    // presence of an x87 floating-point processor.
#if defined(__i386__)||defined(__x86_64)
    u_int16_t flags;
    __asm__("fxam\n\t"
	    "fstsw %%ax"
	    : "=a" (flags) /* output */
	    : "t"  (x)     /* input */
            :              /* clobbered */
            );
    return (flags & 0x4500)==0x0100;
#else
    #error No asm_isnan for this architecture.
#endif    
  }

  template <class FP> inline bool equal_isnan(FP x)
    {
      return x !=x;
    }

  // Here are the public functions, chosen by best timing on Intel
  // Pentium 4.  Other architectures are likely to have different
  // orderings.
  inline bool isnan(float f) { return detail::isnan(f); }

  inline bool isnan(double d) { return equal_isnan(d); }

  inline bool isnan(long double q) { return detail::isnan(q); }

} // namespace edm

#endif
