#ifndef DataFormat_Math_SSEArray_H
#define DataFormat_Math_SSEArray_H


#include "DataFormats/Math/interface/SSEVec.h"
#include<cmath>

#ifdef  CMS_USE_SSE
namespace mathSSE {

// "vertical array"
  template<typename T, size_t S>
  struct ArrayTraits {
  };
  
  template<typename T, size_t S>
  struct ArrayMask {
  };
  
  //FIXME avoid punning...
  template <>
  struct ArrayMask<float, 0> {
    static inline Vec4<float> value() {
      Vec4<float> v; v.setMask(0xffffffff,  0xffffffff,  0xffffffff,  0xffffffff);
      return v;
    }
  };
  template <>
  struct ArrayMask<float,1> {
    static inline Vec4<float> value() {
      Vec4<float> v; v.setMask(0xffffffff,  0x0,  0x0,  0x0);
      return v;
    }
  };
  template <>
  struct ArrayMask<float,2> {
    static inline Vec4<float> value() {
     Vec4<float> v; v.setMask(0xffffffff,  0xffffffff,  0x0,  0x0);
     return v;
    }
  };
  template <>
  struct ArrayMask<float,3> {
    static inline Vec4<float> value() {
      Vec4<float> v; v.setMask(0xffffffff,  0xffffffff,  0xffffffff,  0x0);
      return v;
   }
  };
  
  template <>
  struct ArrayMask<double, 0> {
    static inline Vec2<double> value() {
      Vec2<double> v; v.setMask(0xffffffffffffffffLL,  0xffffffffffffffffLL);
      return v;
    }
  };
  template <>
  struct ArrayMask<double,1> {
    static inline Vec2<double> value() {
      Vec2<double> v; v.setMask(0xffffffffffffffffLL,  0x0LL);
      return v;
    }
  };
  
  
  template<size_t S>
  struct ArrayTraits<float, S> {
    typedef float Scalar;
    typedef Vec4<float> Vec;
    static const size_t size = S;
    static const size_t ssesize = (S+3)/4;
    static const size_t arrsize = 4*ssesize;
    static inline Vec maskLast() { return ArrayMask<Scalar,arrsize-size>::value(); }
    static inline Vec __attribute__((__always_inline__)) mask(Vec v, size_t i) {
      return (i==ssesize-1) ? maskLast()&v : v;
    }
    template <typename F>
    static void loop(F f) {
      for (size_t i=0; i!=ssesize-1;++i)
	f(i, ArrayMask<Scalar,0>::value());
      f(ssesize-1,maskLast());
    }
  };
  
  template<size_t S>
  struct ArrayTraits<double, S> {
    typedef double Scalar;
    typedef Vec2<double> Vec;
    static const size_t size = S;
    static const size_t ssesize = (S+1)/2;
    static const size_t arrsize = 2*ssesize;
    static inline Vec maskLast() { return ArrayMask<Scalar,arrsize-size>::value();}
  };
  
  template<typename T, size_t S>
  union Array {
    typedef ArrayTraits<T,S> Traits;
    typedef typename Traits::Vec Vec;
    typename Vec::nativeType vec[Traits::ssesize];
    T __attribute__ ((aligned(16))) arr[Traits::arrsize];

    Vec operator[]( size_t i) { return vec[i];}
    Vec const & operator[]( size_t i) const{ return reinterpret_cast<Vec const &>(vec[i]);}

  };


}

#endif //  CMS_USE_SSE
#endif // DataFormat_Math_SSEArray_H
