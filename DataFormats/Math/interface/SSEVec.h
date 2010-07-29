#ifndef DataFormat_Math_SSEVec_H
#define DataFormat_Math_SSEVec_H

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

namespace mathSSE {
#ifdef  CMS_USE_SSE
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
  

  // cross (just 3x3) 
  inline __m128 _mm_cross_ps(__m128 v1, __m128 v2) {
    __m128 v3 = _mm_shuffle_ps(v2, v1, _MM_SHUFFLE(3, 0, 2, 2));
    __m128 v4 = _mm_shuffle_ps(v1, v2, _MM_SHUFFLE(3, 1, 0, 1));
    
    __m128 v5 = _mm_mul_ps(v3, v4);
    
    v3 = _mm_shuffle_ps(v1, v2, _MM_SHUFFLE(3, 0, 2, 2));
    v4 = _mm_shuffle_ps(v2, v1, _MM_SHUFFLE(3, 1, 0, 1));
    
    v3 = _mm_mul_ps(v3, v4);
    const  __m128 neg = _mm_set_ps(0.0f,0.0f,-0.0f,0.0f);
    return _mm_xor_ps(_mm_sub_ps(v5, v3), neg);
  }


#endif // CMS_USE_SSE


  template<typename T>
  struct OldVec { T  theX; T  theY; T  theZ; T  theW;}  __attribute__ ((aligned (16)));
  

  template<typename T> union Vec2{
    Vec2() {
      arr[0] = 0; arr[1] = 0;
    }
    Vec2(T f1, T f2) {
      arr[0] = f1; arr[1] = f2;
    }
    explicit Vec2(T f1) {
      arr[0] = f1; arr[1] = f1;
    }
    void set(T f1, T f2) {
      arr[0] = f1; arr[1] = f2;
    }
    Vec2 get1(unsigned int n) const {
      return Vec2(arr[n],arr[n]);
    }

    T __attribute__ ((aligned(16))) arr[2];
  };


  template<typename T> union Vec3{
    Vec3() {
      arr[0] = 0; arr[1] = 0; arr[2] = 0; arr[3]=0;
    }
    Vec3(float f1, float f2, float f3, float f4=0) {
      arr[0] = f1; arr[1] = f2; arr[2] = f3; arr[3]=f4;
    }
    explicit Vec3(float f1) {
      set1(f1);
    }
    void set(float f1, float f2, float f3, float f4=0) {
      arr[0] = f1; arr[1] = f2; arr[2] = f3; arr[3]=f4;
    }
    void set1(float f1) {
      arr[0] = f1; arr[1] = f1; arr[2] = f1; arr[3]=f1;
    }
    Vec3 get1(unsigned int n) const {
      return Vec3(arr[n],arr[n],arr[n],arr[n]);
    }

    T __attribute__ ((aligned(16))) arr[4];
    OldVec<T> o;
  };


#ifdef CMS_USE_SSE

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

    explicit Vec3(float f1) {
      set1(f1);
    }

    Vec3(float f1, float f2, float f3, float f4=0) {
      arr[0] = f1; arr[1] = f2; arr[2] = f3; arr[3]=f4;
    }

    void set(float f1, float f2, float f3, float f4=0) {
      vec = _mm_set_ps(f4, f3, f2, f1);
    }
    void set1(float f1) {
     vec =  _mm_set1_ps(f1);
    }
    Vec3 get1(unsigned int n) const { 
      return _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(n, n, n, n)); 
  }

  };
  
  template<>
  union Vec2<double> {
    __m128d vec;
    double __attribute__ ((aligned(16))) arr[2];
        
    Vec2(__m128d ivec) : vec(ivec) {}
    
    Vec2() {
      vec = _mm_setzero_pd();
    }

    Vec2(double f1, double f2) {
      arr[0] = f1; arr[1] = f2;
    }

    explicit Vec2(double f1) {
      set1(f1);
    }
    
    void set(double f1, double f2) {
      arr[0] = f1; arr[1] = f2;
    }

    void set1(double f1) {
      vec = _mm_set1_pd(f1);
    }

    Vec2 get1(unsigned int n) const {
      return Vec2(arr[n],arr[n]);
    }
   
    double operator[](unsigned int n) const {
      return arr[n];
    }
  };
 

  template<>
  union Vec3<double> {
    __m128d vec[2];
    double __attribute__ ((aligned(16))) arr[4];
    OldVec<double> o;
    
    Vec3(__m128d ivec[]) {
      vec[0] = ivec[0];
      vec[1] = ivec[1];
    }
    
    Vec3(__m128d ivec0, __m128d ivec1) {
      vec[0] = ivec0;
      vec[1] = ivec1;
    }
    
    Vec3() {
      vec[0] = _mm_setzero_pd();
      vec[1] = _mm_setzero_pd();
    }

    explicit Vec3(double f1) {
      set1(f1);
    }

    Vec3(double f1, double f2, double f3, double f4=0) {
      arr[0] = f1; arr[1] = f2; arr[2] = f3; arr[3]=f4;
    }
    
    Vec3(OldVec<double> const & ivec) : o(ivec) {}

    void set(double f1, double f2, double f3, double f4=0) {
      arr[0] = f1; arr[1] = f2; arr[2] = f3; arr[3]=f4;
    }

    void set1(double f1) {
      vec[0] = vec[1]= _mm_set1_pd(f1);
    }


    Vec3 get1(unsigned int n) const {
      return Vec3(arr[n],arr[n],arr[n],arr[n]);
    }

    Vec2<double> xy() const { return vec[0];}
    Vec2<double> zw() const { return vec[1];}

  };

#endif // CMS_USE_SSE
  
  typedef Vec3<float> Vec3F;
  typedef Vec3<float> Vec4F;
  typedef Vec2<double> Vec2D;
  typedef Vec3<double> Vec3D;
  typedef Vec3<double> Vec4D;

}

#ifdef CMS_USE_SSE


//float op

inline float dot(mathSSE::Vec3F a, mathSSE::Vec3F b) {
  using  mathSSE::_mm_dot_ps;
  float s;
  _mm_store_ss(&s,_mm_dot_ps(a.vec,b.vec));
  return s;
}

inline mathSSE::Vec3F cross(mathSSE::Vec3F a, mathSSE::Vec3F b) {
  using  mathSSE::_mm_cross_ps;
  return _mm_cross_ps(a.vec,b.vec);
}


inline bool operator==(mathSSE::Vec3F a, mathSSE::Vec3F b) {
  return _mm_movemask_ps(_mm_cmpeq_ps(a.vec,b.vec))==0xf;
}

inline mathSSE::Vec3F operator-(mathSSE::Vec3F a) {
  const __m128 neg = _mm_set_ps ( -0.0 , -0.0 , -0.0, -0.0);
  return _mm_xor_ps(a.vec,neg);
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


// double op 2d
inline mathSSE::Vec2D operator-(mathSSE::Vec2D a) {
  const __m128d neg = _mm_set_pd ( -0.0 , -0.0);
  return _mm_xor_pd(a.vec,neg);
}

inline mathSSE::Vec2D operator+(mathSSE::Vec2D a, mathSSE::Vec2D b) {
  return  _mm_add_pd(a.vec,b.vec);
}

inline mathSSE::Vec2D operator-(mathSSE::Vec2D a, mathSSE::Vec2D b) {
  return  _mm_sub_pd(a.vec,b.vec);
}

inline mathSSE::Vec2D operator*(mathSSE::Vec2D a, mathSSE::Vec2D b) {
  return  _mm_mul_pd(a.vec,b.vec);
}

inline mathSSE::Vec2D operator/(mathSSE::Vec2D a, mathSSE::Vec2D b) {
  return  _mm_div_pd(a.vec,b.vec);
}

inline mathSSE::Vec2D operator*(double a, mathSSE::Vec2D b) {
  return  _mm_mul_pd(_mm_set1_pd(a),b.vec);
}

inline mathSSE::Vec2D operator*(mathSSE::Vec2D b,double a) {
  return  _mm_mul_pd(_mm_set1_pd(a),b.vec);
}

inline double dot(mathSSE::Vec2D a, mathSSE::Vec2D b)  __attribute__((always_inline)) __attribute__ ((pure));

inline double dot(mathSSE::Vec2D a, mathSSE::Vec2D b){
  __m128d res = _mm_mul_pd ( a.vec, b.vec);
  res = _mm_add_sd (  _mm_shuffle_pd ( res , res, 1 ), res );
  double s;
  _mm_store_sd(&s,res);
  return s;
}

inline double cross(mathSSE::Vec2D a, mathSSE::Vec2D b)  __attribute__((always_inline)) __attribute__ ((pure));

inline double cross(mathSSE::Vec2D a, mathSSE::Vec2D b) {
  __m128d res =  _mm_shuffle_pd ( b.vec, b.vec, 1);
  res = _mm_mul_pd (  a.vec , res );
  res = _mm_sub_sd (res, _mm_shuffle_pd ( res , res, 1 ));
  double s;
  _mm_store_sd(&s,res);
  return s;
}


// double op 3d

inline bool operator==(mathSSE::Vec3D a, mathSSE::Vec3D b) {
  return 
    _mm_movemask_pd(_mm_cmpeq_pd(a.vec[0],b.vec[0]))==0x3 && 
    _mm_movemask_pd(_mm_cmpeq_pd(a.vec[1],b.vec[1]))==0x3 ;
}

inline mathSSE::Vec3D operator-(mathSSE::Vec3D a) {
  const __m128d neg = _mm_set_pd ( -0.0 , -0.0);
  return mathSSE::Vec3D(_mm_xor_pd(a.vec[0],neg),_mm_xor_pd(a.vec[1],neg));
}


inline mathSSE::Vec3D operator+(mathSSE::Vec3D a, mathSSE::Vec3D b) {
  return  mathSSE::Vec3D(_mm_add_pd(a.vec[0],b.vec[0]),_mm_add_pd(a.vec[1],b.vec[1]));
}
inline mathSSE::Vec3D operator-(mathSSE::Vec3D a, mathSSE::Vec3D b) {
  return  mathSSE::Vec3D(_mm_sub_pd(a.vec[0],b.vec[0]),_mm_sub_pd(a.vec[1],b.vec[1]));
}
inline mathSSE::Vec3D operator*(mathSSE::Vec3D a, mathSSE::Vec3D b) {
  return  mathSSE::Vec3D(_mm_mul_pd(a.vec[0],b.vec[0]),_mm_mul_pd(a.vec[1],b.vec[1]));
}
inline mathSSE::Vec3D operator/(mathSSE::Vec3D a, mathSSE::Vec3D b) {
  return  mathSSE::Vec3D(_mm_div_pd(a.vec[0],b.vec[0]),_mm_div_pd(a.vec[1],b.vec[1]));
}

inline mathSSE::Vec3D operator*(double a, mathSSE::Vec3D b) {
  __m128d res = _mm_set1_pd(a);
  return  mathSSE::Vec3D(_mm_mul_pd(res,b.vec[0]),_mm_mul_pd(res,b.vec[1]));
}

inline mathSSE::Vec3D operator*(mathSSE::Vec3D b, double a) {
  __m128d res = _mm_set1_pd(a);
  return  mathSSE::Vec3D(_mm_mul_pd(res,b.vec[0]),_mm_mul_pd(res,b.vec[1]));
}



inline double dot(mathSSE::Vec3D a, mathSSE::Vec3D b) __attribute__((always_inline)) __attribute__ ((pure));

inline double dot(mathSSE::Vec3D a, mathSSE::Vec3D b) {
  __m128d res = _mm_add_sd ( _mm_mul_pd ( a.vec[0], b.vec[0]),
			     _mm_mul_sd ( a.vec[1], b.vec[1]) 
			     );
  res = _mm_add_sd ( _mm_unpackhi_pd ( res , res ), res );
  double s;
  _mm_store_sd(&s,res);
  return s;
}

inline mathSSE::Vec3D cross(mathSSE::Vec3D a, mathSSE::Vec3D b) __attribute__((always_inline)) __attribute__ ((pure));
 
inline mathSSE::Vec3D cross(mathSSE::Vec3D a, mathSSE::Vec3D b) {
  const __m128d neg = _mm_set_pd ( 0.0 , -0.0 );
  // lh .z * rh .x , lh .z * rh .y
  __m128d l1 = _mm_mul_pd ( _mm_unpacklo_pd ( a.vec[1] , a.vec[1] ), b.vec[0] );
  // rh .z * lh .x , rh .z * lh .y
  __m128d l2 = _mm_mul_pd ( _mm_unpacklo_pd (  b.vec[1],  b.vec[1] ),  a.vec[0] );
  __m128d m1 = _mm_sub_pd ( l1 , l2 ); // l1 - l2
  m1 = _mm_shuffle_pd ( m1 , m1 , 1 ); // switch the elements
  m1 = _mm_xor_pd ( m1 , neg ); // change the sign of the first element
  // lh .x * rh .y , lh .y * rh .x
  l1 = _mm_mul_pd (  a.vec[0] , _mm_shuffle_pd (  b.vec[0] ,  b.vec[0] , 1 ) );
  // lh .x * rh .y - lh .y * rh .x
  __m128d m2 = _mm_sub_sd ( l1 , _mm_unpackhi_pd ( l1 , l1 ) );

  return  mathSSE::Vec3D( m1 , m2 );
}



// sqrt
namespace mathSSE {
  template<typename T> inline T sqrt(T t) { return std::sqrt(t);}
  template<> inline Vec3F sqrt(Vec3F v) { return _mm_sqrt_ps(v.vec);}
  template<> inline Vec2D sqrt(Vec2D v) { return _mm_sqrt_pd(v.vec);}
  template<> inline Vec3D sqrt(Vec3D v) { 
    return Vec3D(_mm_sqrt_pd(v.vec[0]),_mm_sqrt_pd(v.vec[1]));
  }
}

#endif // CMS_USE_SSE


#include <iosfwd>
std::ostream & operator<<(std::ostream & out, mathSSE::Vec2D const & v);
std::ostream & operator<<(std::ostream & out, mathSSE::Vec3F const & v);
std::ostream & operator<<(std::ostream & out, mathSSE::Vec3D const & v);

#endif // DataFormat_Math_SSEVec_H
