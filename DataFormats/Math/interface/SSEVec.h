#ifndef DataFormat_Math_SSEVec_H
#define DataFormat_Math_SSEVec_H

#if defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ > 4)
#include <x86intrin.h>

#else

#include <mmintrin.h>
#include <emmintrin.h>
#ifdef __SSE3__
#include <pmmintrin.h>
#endif
#ifdef __SSE4_1__
#include <smmintrin.h>
#endif

#endif



namespace mathSSE {

  //dot
  inline __m128 _mm_dot_ps(__m128 v1, __m128 v2) {
#ifdef __SSE4_1__
    return _mm_dp_ps(v1, v2, 0xff);
#else
    __m128 mul = _mm_mul_ps(v1, v2);
#ifdef __SSE3__
    mul = _mm_hadd_ps(mul,mul);
    return _mm_hadd_ps(mul,mul);
#else
    __m128 swp = _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(1, 0, 3, 2));
    mul = _mm_add_ps(mul, swp);
    swp = _mm_shuffle_ps(mul, mul, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_add_ps(mul, swp);
#endif
#endif
  }
  
  // almost cross (just 3x3 and [1] has the wrong sign....) 
  inline __m128 _mm_cross_ps(__m128 v1, __m128 v2) {
    __m128 v3 = _mm_shuffle_ps(v2, v1, _MM_SHUFFLE(3, 0, 2, 2));
    __m128 v4 = _mm_shuffle_ps(v1, v2, _MM_SHUFFLE(3, 1, 0, 1));
    
    __m128 v5 = _mm_mul_ps(v3, v4);
    
    v3 = _mm_shuffle_ps(v1, v2, _MM_SHUFFLE(3, 0, 2, 2));
    v4 = _mm_shuffle_ps(v2, v1, _MM_SHUFFLE(3, 1, 0, 1));
    
    v3 = _mm_mul_ps(v3, v4);
    return _mm_sub_ps(v5, v3);
  }


  template<typename T>
  struct OldVec { T  theX; T  theY; T  theZ; T  theW;}  __attribute__ ((aligned (16)));
  
  template<typename T> union Vec3{
    Vec3() {
      arr[0] = 0; arr[1] = 0; arr[2] = 0; arr[3]=0;
    }
    Vec3(float f1, float f2, float f3) {
      arr[0] = f1; arr[1] = f2; arr[2] = f3; arr[3]=0;
    }
    T __attribute__ ((aligned(16))) arr[4];
    OldVec<T> o;
  };

  template<>
  union Vec3<float> {
    __m128 vec;
    float __attribute__ ((aligned(16))) arr[4];
    OldVec<float> o;
    
    Vec3(__m128 ivec) : vec(ivec) {}

    Vec3(OldVec<float> const & ivec) : o(ivec) {}
    
    Vec3() {
      vec = _mm_setzero_ps();
    }
    Vec3(float f1, float f2, float f3) {
      arr[0] = f1; arr[1] = f2; arr[2] = f3; arr[3]=0;
    }
  };
  
  typedef Vec3<float> Vec3F;

  template<>
  union Vec3<double> {
    __m128d vec[2];
    double __attribute__ ((aligned(16))) arr[4];
    OldVec<double> o;
    
    Vec3(__m128d ivec[]) {
      vec[0] = ivec[0];
      vec[1] = ivec[1];
    }
    
    Vec3() {
      vec[0] = _mm_setzero_pd();
      vec[1] = _mm_setzero_pd();
    }

    Vec3(double f1, double f2, double f3) {
      arr[0] = f1; arr[1] = f2; arr[2] = f3; arr[3]=0;
    }
    
    Vec3(OldVec<double> const & ivec) : o(ivec) {}

  };
  
  typedef Vec3<double> Vec3D;

}


inline float dot(mathSSE::Vec3F a, mathSSE::Vec3F b) {
  using  mathSSE::_mm_dot_ps;
  float s;
  _mm_store_ss(&s,_mm_dot_ps(a.vec,b.vec));
  return s;
}

inline mathSSE::Vec3F cross(mathSSE::Vec3F a, mathSSE::Vec3F b) {
  using  mathSSE::_mm_cross_ps;
  mathSSE::Vec3F res(_mm_cross_ps(a.vec,b.vec));
  res.arr[1] *= -1.f;
  return res;
}


inline mathSSE::Vec3F operator+(mathSSE::Vec3F a, mathSSE::Vec3F b) {
  return  _mm_add_ps(a.vec,b.vec);
}

inline mathSSE::Vec3F operator-(mathSSE::Vec3F a, mathSSE::Vec3F b) {
  return  _mm_sub_ps(a.vec,b.vec);
}

inline mathSSE::Vec3F operator*(mathSSE::Vec3F a, mathSSE::Vec3F b) {
  return  _mm_mul_ps(a.vec,b.vec);
}

inline mathSSE::Vec3F operator/(mathSSE::Vec3F a, mathSSE::Vec3F b) {
  return  _mm_div_ps(a.vec,b.vec);
}

inline mathSSE::Vec3F operator*(float a, mathSSE::Vec3F b) {
  return  _mm_mul_ps(_mm_set1_ps(a),b.vec);
}

inline mathSSE::Vec3F operator*(mathSSE::Vec3F b,float a) {
  return  _mm_mul_ps(_mm_set1_ps(a),b.vec);
}


#include <iosfwd>
std::ostream & operator<<(std::ostream & out, mathSSE::Vec3F const & v);

#endif // DataFormat_Math_SSEVec_H
