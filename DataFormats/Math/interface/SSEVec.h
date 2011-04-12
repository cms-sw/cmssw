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
  template<typename T> inline T sqrt(T t) { return std::sqrt(t);}
}

namespace mathSSE {
  //
  template<typename T> inline bool samesign(T rh, T lh);

  template<>
  inline bool
  __attribute__((always_inline)) __attribute__ ((pure)) samesign<int>(int rh, int lh) {
    int const mask= 0x80000000;
    return ((rh^lh)&mask) == 0;
  }

  template<>
  inline bool
  __attribute__((always_inline)) __attribute__ ((pure)) samesign<long long>(long long rh, long long lh) {
    long long const mask= 0x8000000000000000LL;
    return ((rh^lh)&mask) == 0;
  }

  template<>
  inline bool
  __attribute__((always_inline)) __attribute__ ((pure)) samesign<float>(float rh, float lh) {
    union { int i; float f; } a, b;
    a.f=rh; b.f=lh;
    return samesign<int>(a.i,b.i);
  }

  template<>
  inline bool
  __attribute__((always_inline)) __attribute__ ((pure)) samesign<double>(double rh, double lh) {
    union { long long i; double f; } a, b;
    a.f=rh; b.f=lh;
    return samesign<long long>(a.i,b.i);
  }
}


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

    T & operator[](unsigned int n) {
      return arr[n];
    }

    T operator[](unsigned int n) const {
      return arr[n];
    }


    T __attribute__ ((aligned(16))) arr[2];
  };


  template<typename T> union Vec4{
    Vec4() {
      arr[0] = 0; arr[1] = 0; arr[2] = 0; arr[3]=0;
    }
    Vec4(float f1, float f2, float f3, float f4=0) {
      arr[0] = f1; arr[1] = f2; arr[2] = f3; arr[3]=f4;
    }
    explicit Vec4(float f1) {
      set1(f1);
    }
    void set(float f1, float f2, float f3, float f4=0) {
      arr[0] = f1; arr[1] = f2; arr[2] = f3; arr[3]=f4;
    }
    void set1(float f1) {
      arr[0] = f1; arr[1] = f1; arr[2] = f1; arr[3]=f1;
    }
    Vec4 get1(unsigned int n) const {
      return Vec4(arr[n],arr[n],arr[n],arr[n]);
    }

    Vec2<T> xy() const { return  Vec2<T>(arr[0],arr[1]);}
    Vec2<T> zw() const { return  Vec2<T>(arr[2],arr[3]);}



    T __attribute__ ((aligned(16))) arr[4];
    OldVec<T> o;
  };


#ifdef CMS_USE_SSE

  template<>
  union Vec4<float> {
    typedef  __m128 nativeType;
    __m128 vec;
    float __attribute__ ((aligned(16))) arr[4];
    OldVec<float> o;
    
    Vec4(__m128 ivec) : vec(ivec) {}

    Vec4(OldVec<float> const & ivec) : o(ivec) {}
    
    Vec4() {
      vec = _mm_setzero_ps();
    }

    explicit Vec4(float f1) {
      set1(f1);
    }

    Vec4(float f1, float f2, float f3, float f4=0) {
      arr[0] = f1; arr[1] = f2; arr[2] = f3; arr[3]=f4;
    }

    void set(float f1, float f2, float f3, float f4=0) {
      vec = _mm_set_ps(f4, f3, f2, f1);
    }
    void set1(float f1) {
     vec =  _mm_set1_ps(f1);
    }

    Vec4 get1(unsigned int n) const { 
      return _mm_shuffle_ps(vec, vec, _MM_SHUFFLE(n, n, n, n)); 
    }

    float & operator[](unsigned int n) {
      return arr[n];
    }

    float operator[](unsigned int n) const {
      return arr[n];
    }
    
    Vec2<float> xy() const { return  Vec2<float>(arr[0],arr[1]);}
    Vec2<float> zw() const { return  Vec2<float>(arr[2],arr[3]);}

  };
  
  template<>
  union Vec2<double> {
    typedef  __m128d nativeType;
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
  union Vec4<double> {
    __m128d vec[2];
    double __attribute__ ((aligned(16))) arr[4];
    OldVec<double> o;
    
    Vec4(__m128d ivec[]) {
      vec[0] = ivec[0];
      vec[1] = ivec[1];
    }
    
    Vec4(__m128d ivec0, __m128d ivec1) {
      vec[0] = ivec0;
      vec[1] = ivec1;
    }
    
    Vec4() {
      vec[0] = _mm_setzero_pd();
      vec[1] = _mm_setzero_pd();
    }

    explicit Vec4(double f1) {
      set1(f1);
    }

    Vec4(double f1, double f2, double f3, double f4=0) {
      arr[0] = f1; arr[1] = f2; arr[2] = f3; arr[3]=f4;
    }
    
   Vec4( Vec2<double> ivec0,   Vec2<double> ivec1) {
      vec[0] = ivec0.vec;
      vec[1] = ivec1.vec;
    }
    
    Vec4( Vec2<double> ivec0,  double f3, double f4=0) {
      vec[0] = ivec0.vec;
      arr[2] = f3; arr[3] = f4;
    }

   Vec4( Vec2<double> ivec0) {
      vec[0] = ivec0.vec;
      vec[1] =  _mm_setzero_pd();
    }


    Vec4(OldVec<double> const & ivec) : o(ivec) {}

    void set(double f1, double f2, double f3, double f4=0) {
      arr[0] = f1; arr[1] = f2; arr[2] = f3; arr[3]=f4;
    }

    void set1(double f1) {
      vec[0] = vec[1]= _mm_set1_pd(f1);
    }


    Vec4 get1(unsigned int n) const {
      return Vec4(arr[n],arr[n],arr[n],arr[n]);
    }

    double & operator[](unsigned int n) {
      return arr[n];
    }

    double operator[](unsigned int n) const {
      return arr[n];
    }
  
    Vec2<double> xy() const { return vec[0];}
    Vec2<double> zw() const { return vec[1];}

  };

#endif // CMS_USE_SSE
  
  typedef Vec4<float> Vec4F;
  typedef Vec4<float> Vec3F;
  typedef Vec2<double> Vec2D;
  typedef Vec4<double> Vec3D;
  typedef Vec4<double> Vec4D;

  template<typename T>
  struct As3D {
    Vec4<T> const & v;
    As3D(Vec4<T> const &iv ) : v(iv){}
  };

  template<typename T>
  inline As3D<T> as3D(Vec4<T> const &v ) { return v;}

}

#ifdef CMS_USE_SSE


//float op

inline float dot(mathSSE::Vec4F a, mathSSE::Vec4F b) {
  using  mathSSE::_mm_dot_ps;
  float s;
  _mm_store_ss(&s,_mm_dot_ps(a.vec,b.vec));
  return s;
}

inline mathSSE::Vec4F cross(mathSSE::Vec4F a, mathSSE::Vec4F b) {
  using  mathSSE::_mm_cross_ps;
  return _mm_cross_ps(a.vec,b.vec);
}


inline bool operator==(mathSSE::Vec4F a, mathSSE::Vec4F b) {
  return _mm_movemask_ps(_mm_cmpeq_ps(a.vec,b.vec))==0xf;
}

inline mathSSE::Vec4F cmpeq(mathSSE::Vec4F a, mathSSE::Vec4F b) {
  return _mm_cmpeq_ps(a.vec,b.vec);
}

inline mathSSE::Vec4F cmpgt(mathSSE::Vec4F a, mathSSE::Vec4F b) {
  return _mm_cmpgt_ps(a.vec,b.vec);
}

#ifdef __SSE3__
inline mathSSE::Vec4F hadd(mathSSE::Vec4F a, mathSSE::Vec4F b) {
  return _mm_hadd_ps(a.vec,b.vec);
}
#endif


inline mathSSE::Vec4F operator-(mathSSE::Vec4F a) {
  const __m128 neg = _mm_set_ps ( -0.0 , -0.0 , -0.0, -0.0);
  return _mm_xor_ps(a.vec,neg);
}

inline mathSSE::Vec4F operator&(mathSSE::Vec4F a, mathSSE::Vec4F b) {
  return  _mm_and_ps(a.vec,b.vec);
}
inline mathSSE::Vec4F operator|(mathSSE::Vec4F a, mathSSE::Vec4F b) {
  return  _mm_or_ps(a.vec,b.vec);
}
inline mathSSE::Vec4F operator^(mathSSE::Vec4F a, mathSSE::Vec4F b) {
  return  _mm_xor_ps(a.vec,b.vec);
}
inline mathSSE::Vec4F andnot(mathSSE::Vec4F a, mathSSE::Vec4F b) {
  return  _mm_andnot_ps(a.vec,b.vec);
}


inline mathSSE::Vec4F operator+(mathSSE::Vec4F a, mathSSE::Vec4F b) {
  return  _mm_add_ps(a.vec,b.vec);
}

inline mathSSE::Vec4F operator-(mathSSE::Vec4F a, mathSSE::Vec4F b) {
  return  _mm_sub_ps(a.vec,b.vec);
}

inline mathSSE::Vec4F operator*(mathSSE::Vec4F a, mathSSE::Vec4F b) {
  return  _mm_mul_ps(a.vec,b.vec);
}

inline mathSSE::Vec4F operator/(mathSSE::Vec4F a, mathSSE::Vec4F b) {
  return  _mm_div_ps(a.vec,b.vec);
}

inline mathSSE::Vec4F operator*(float a, mathSSE::Vec4F b) {
  return  _mm_mul_ps(_mm_set1_ps(a),b.vec);
}

inline mathSSE::Vec4F operator*(mathSSE::Vec4F b,float a) {
  return  _mm_mul_ps(_mm_set1_ps(a),b.vec);
}


// double op 2d
inline mathSSE::Vec2D operator-(mathSSE::Vec2D a) {
  const __m128d neg = _mm_set_pd ( -0.0 , -0.0);
  return _mm_xor_pd(a.vec,neg);
}


inline mathSSE::Vec2D operator&(mathSSE::Vec2D a, mathSSE::Vec2D b) {
  return  _mm_and_pd(a.vec,b.vec);
}
inline mathSSE::Vec2D operator|(mathSSE::Vec2D a, mathSSE::Vec2D b) {
  return  _mm_or_pd(a.vec,b.vec);
}
inline mathSSE::Vec2D operator^(mathSSE::Vec2D a, mathSSE::Vec2D b) {
  return  _mm_xor_pd(a.vec,b.vec);
}
inline mathSSE::Vec2D andnot(mathSSE::Vec2D a, mathSSE::Vec2D b) {
  return  _mm_andnot_pd(a.vec,b.vec);
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

inline bool operator==(mathSSE::Vec4D a, mathSSE::Vec4D b) {
  return 
    _mm_movemask_pd(_mm_cmpeq_pd(a.vec[0],b.vec[0]))==0x3 && 
    _mm_movemask_pd(_mm_cmpeq_pd(a.vec[1],b.vec[1]))==0x3 ;
}

inline mathSSE::Vec4D operator-(mathSSE::Vec4D a) {
  const __m128d neg = _mm_set_pd ( -0.0 , -0.0);
  return mathSSE::Vec4D(_mm_xor_pd(a.vec[0],neg),_mm_xor_pd(a.vec[1],neg));
}


inline mathSSE::Vec4D operator+(mathSSE::Vec4D a, mathSSE::Vec4D b) {
  return  mathSSE::Vec4D(_mm_add_pd(a.vec[0],b.vec[0]),_mm_add_pd(a.vec[1],b.vec[1]));
}
inline mathSSE::Vec4D operator-(mathSSE::Vec4D a, mathSSE::Vec4D b) {
  return  mathSSE::Vec4D(_mm_sub_pd(a.vec[0],b.vec[0]),_mm_sub_pd(a.vec[1],b.vec[1]));
}
inline mathSSE::Vec4D operator*(mathSSE::Vec4D a, mathSSE::Vec4D b) {
  return  mathSSE::Vec4D(_mm_mul_pd(a.vec[0],b.vec[0]),_mm_mul_pd(a.vec[1],b.vec[1]));
}
inline mathSSE::Vec4D operator/(mathSSE::Vec4D a, mathSSE::Vec4D b) {
  return  mathSSE::Vec4D(_mm_div_pd(a.vec[0],b.vec[0]),_mm_div_pd(a.vec[1],b.vec[1]));
}

inline mathSSE::Vec4D operator*(double a, mathSSE::Vec4D b) {
  __m128d res = _mm_set1_pd(a);
  return  mathSSE::Vec4D(_mm_mul_pd(res,b.vec[0]),_mm_mul_pd(res,b.vec[1]));
}

inline mathSSE::Vec4D operator*(mathSSE::Vec4D b, double a) {
  __m128d res = _mm_set1_pd(a);
  return  mathSSE::Vec4D(_mm_mul_pd(res,b.vec[0]),_mm_mul_pd(res,b.vec[1]));
}



inline double dot(mathSSE::Vec4D a, mathSSE::Vec4D b) __attribute__((always_inline)) __attribute__ ((pure));

inline double dot(mathSSE::Vec4D a, mathSSE::Vec4D b) {
  __m128d res = _mm_add_sd ( _mm_mul_pd ( a.vec[0], b.vec[0]),
			     _mm_mul_sd ( a.vec[1], b.vec[1]) 
			     );
  res = _mm_add_sd ( _mm_unpackhi_pd ( res , res ), res );
  double s;
  _mm_store_sd(&s,res);
  return s;
}

inline mathSSE::Vec4D cross(mathSSE::Vec4D a, mathSSE::Vec4D b) __attribute__((always_inline)) __attribute__ ((pure));
 
inline mathSSE::Vec4D cross(mathSSE::Vec4D a, mathSSE::Vec4D b) {
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

  return  mathSSE::Vec4D( m1 , m2 );
}



// sqrt
namespace mathSSE {
  template<> inline Vec4F sqrt(Vec4F v) { return _mm_sqrt_ps(v.vec);}
  template<> inline Vec2D sqrt(Vec2D v) { return _mm_sqrt_pd(v.vec);}
  template<> inline Vec4D sqrt(Vec4D v) { 
    return Vec4D(_mm_sqrt_pd(v.vec[0]),_mm_sqrt_pd(v.vec[1]));
  }
}

// chephes func
#include "DataFormats/Math/interface/sse_mathfun.h"
namespace mathSSE {
  inline Vec4F log(Vec4F v) { return log_ps(v.vec);}
  inline Vec4F exp(Vec4F v) { return exp_ps(v.vec);}
  inline Vec4F sin(Vec4F v) { return sin_ps(v.vec);}
  inline Vec4F cos(Vec4F v) { return cos_ps(v.vec);}
  inline void sincos(Vec4F v, Vec4F & s, Vec4F & c) { sincos_ps(v.vec,&s.vec, &c.vec);}

  inline float log(float f) { float s; _mm_store_ss(&s,log_ps(_mm_load_ss(&f))); return s;}
  inline float exp(float f) { float s; _mm_store_ss(&s,exp_ps(_mm_load_ss(&f))); return s;}
  inline float sin(float f) { float s; _mm_store_ss(&s,sin_ps(_mm_load_ss(&f))); return s;}
  inline float cos(float f) { float s; _mm_store_ss(&s,log_ps(_mm_load_ss(&f))); return s;}
  inline void sincos(float f, float & s, float & c) { 
    __m128 vs, vc; 
    sincos_ps(_mm_load_ss(&f),&vs, &vc);   
    _mm_store_ss(&s,vs);_mm_store_ss(&c,vc);   
  }
}
#endif // CMS_USE_SSE


#include <iosfwd>
std::ostream & operator<<(std::ostream & out, mathSSE::Vec2D const & v);
std::ostream & operator<<(std::ostream & out, mathSSE::Vec4F const & v);
std::ostream & operator<<(std::ostream & out, mathSSE::Vec4D const & v);

std::ostream & operator<<(std::ostream & out, mathSSE::As3D<float> const & v);
std::ostream & operator<<(std::ostream & out, mathSSE::As3D<double> const & v);


#endif // DataFormat_Math_SSEVec_H
