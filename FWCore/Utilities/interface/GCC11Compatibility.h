#ifndef FWCORE_GCC11COMPATIBILITY_H
#define FWCORE_GCC11COMPATIBILITY_H
/*
 * set of macro to control the level of comaptibility with c++11 standard
 * includes also other macro more specific to gcc

 */

#include "FWCore/Utilities/interface/Visibility.h"
#include "FWCore/Utilities/interface/Likely.h"


#if defined(__REFLEX__) || defined(__CINT__)
  #define CMS_NOCXX11
#endif
#if !GCC_PREREQUISITE(4,6,0) && !defined(__clang__)
  #define CMS_NOCXX11
#endif

#ifndef CMS_NOCXX11
  #if GCC_PREREQUISITE(4,7,0)
     #define GCC11_FINAL final
     #define GCC11_OVERRIDE override
  #elif __clang__
     // FIXME: do not know how to query for final support. It's there in any
     //        clang version we will ever use, so it's fine.
     #define GCC11_FINAL final
     #if __has_feature(cxx_override_control)
      #define GCC11_OVERRIDE override
     #endif
  #else
     #define GCC11_FINAL
     #define GCC11_OVERRIDE
  #endif // gcc 4.7
#else
  #define constexpr
  #define noexcept
  #define nullptr 0
  #define GCC11_FINAL
  #define GCC11_OVERRIDE
#endif // NOCXX11

#endif  // FWCORE_GCC11COMPATIBILITY_H
