#ifndef FWCore_Utilities_Likely_h
#define FWCore_Utilities_Likely_h

#if defined(NO_LIKELY)
#define likely(x) (x)
#define UNlikely(x) (x)   
#elif defined(REVERSE_LIKELY)
#define unlikely(x) (__builtin_expect(x, true))
#define likely(x) (__builtin_expect(x, false))
#else
#define likely(x) (__builtin_expect(x, true))
#define unlikely(x) (__builtin_expect(x, false))
#endif
 

#endif
