#ifndef DataFormat_Math_SSEArray_H
#define DataFormat_Math_SSEArray_H

#if defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ > 4)
#include <x86intrin.h>
#define CMS_USE_SSE

#else

#ifdef __SSE2__
#define CMS_USE_SSE

#include <mmintrin.h>
#include <emmintrin.h>
#endif
#ifdef __SSE3__
#include <pmmintrin.h>
#endif
#ifdef __SSE4_1__
#include <smmintrin.h>
#endif

#endif

#include<cmath>

#ifdef  CMS_USE_SSE
namespace mathSSE {

// "vertical array"
  template<typename T, size_t S>
  struct Sizes {
  };

  template<size_t S>
  struct Sizes<float, S> {
    typedef __m128 Vec;
    static const size_t size = S;
    static const size_t ssesize = (S+3)/4;
    static const size_t arrsize = 4*ssesize;
  };
  
  template<size_t S>
  struct Sizes<double, S> {
    typedef __m128d Vec;
    static const size_t size = S;
    static const size_t ssesize = (S+1)/2;
    static const size_t arrsize = 2*ssesize;
  };
  
  template<typename T, size_t S>
  union Array {
    typedef Sizes<T,S> Size;
    typedef typename Size::Vec Vec;
    Vec vec[Size::ssesize];
    T __attribute__ ((aligned(16))) arr[Size::arrsize];
  };


}

#endif //  CMS_USE_SSE
#endif // DataFormat_Math_SSEArray_H
