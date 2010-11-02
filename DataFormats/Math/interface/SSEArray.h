#ifndef DataFormat_Math_SSEArray_H
#define DataFormat_Math_SSEArray_H


#include "DataFormats/Math/interface/SSEVec.h"
#include<cmath>

#ifdef  CMS_USE_SSE
namespace mathSSE {

// "vertical array"
  template<typename T, size_t S>
  struct Sizes {
  };

  template<typename T, size_t S>
  struct Mask {
  };
  
  struct Mask<float, 0> {
    static Vec4<float> mask(_mm_cmpeq_ps(_mm_setzero_ps(),_mm_setzero_ps()));
  };
  struct Mask<float,1> {
    static Vec4<float> mask(_mm_cmpeq_ps(_mm_setzero_ps(),Vec4<float>(0,1,1,1)));
  };
  struct Mask<float,2> {
    static Vec4<float> mask(_mm_cmpeq_ps(_mm_setzero_ps(),Vec4<float>(0,0,1,1)));
  };
  struct Mask<float,2> {
    static Vec4<float> mask(_mm_cmpeq_ps(_mm_setzero_ps(),Vec4<float>(0,0,0,1)));
  };

  struct Mask<double, 0> {
    static Vec4<double> mask(_mm_cmpeq_ps(_mm_setzero_ps(),_mm_setzero_ps()));
  };
  struct Mask<double,1> {
    static Vec4<double> mask(_mm_cmpeq_ps(_mm_setzero_ps(),Vec4<double>(0,1)));
  };
  
  
  template<size_t S>
  struct Sizes<float, S> {
    typedef float Scalar;
    typedef Vec4<float> Vec;
    static const size_t size = S;
    static const size_t ssesize = (S+3)/4;
    static const size_t arrsize = 4*ssesize;
    static const maskLast = Mask<Scalar,arrsize-size>::mask;
    template <typename F>
    static void loop(F f) {
      for (size_t i=0; i!=ssesize-1;++i)
	f(i, Mask<Scalar,0>::mask);
      f(ssesize-1,maskLast);
    }
  };
  
  template<size_t S>
  struct Sizes<double, S> {
    typedef double Scalar;
    typedef Vec4<double> Vec;
    static const size_t size = S;
    static const size_t ssesize = (S+1)/2;
    static const size_t arrsize = 2*ssesize;
    static const maskLast = Mask<Scalar,arrsize-size>;    
  };
  
  template<typename T, size_t S>
  union Array {
    typedef Sizes<T,S> Size;
    typedef typename Size::Vec Vec;
    Vec vec[Size::ssesize];
    T __attribute__ ((aligned(16))) arr[Size::arrsize];

    Vec & operator[]( size_t i) { return vec[i];}
    Vec const & operator[]( size_t i) const{ return vec[i];}

  };


}

#endif //  CMS_USE_SSE
#endif // DataFormat_Math_SSEArray_H
