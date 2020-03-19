#ifndef FWCore_Utilities_Likely_h
#define FWCore_Utilities_Likely_h
#include "FWCore/Utilities/interface/GCCPrerequisite.h"

#if GCC_PREREQUISITE(3, 0, 0)

#if defined(NO_LIKELY)
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#elif defined(REVERSE_LIKELY)
#define UNLIKELY(x) (__builtin_expect(x, true))
#define LIKELY(x) (__builtin_expect(x, false))
#else
#define LIKELY(x) (__builtin_expect(x, true))
#define UNLIKELY(x) (__builtin_expect(x, false))
#endif

#else
#define NO_LIKELY
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

#endif
