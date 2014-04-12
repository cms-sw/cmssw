#ifndef DataFormats_Math_FastMath_h
#define DataFormats_Math_FastMath_h
// faster function will a limited precision

#include<cmath>
#include<utility>
#ifdef __SSE2__
# include <emmintrin.h>
#endif
namespace fastmath {
  inline float invSqrt( float in ) {
#ifndef __SSE2__
    return 1.f/std::sqrt(in);
#else
    float out;
    _mm_store_ss( &out, _mm_rsqrt_ss( _mm_load_ss( &in ) ) ); // compiles to movss, rsqrtss, movss
    // return out; // already good enough!
    return out * (1.5f - 0.5f * in * out * out); // One (more?) round of Newton's method
#endif
  }

  inline double invSqrt(double in ) {
    return 1./std::sqrt(in);
  }
  
}
  
namespace fastmath_details {
  const double _2pi = (2.0 * 3.1415926535897932384626434);
  const float _2pif = float(_2pi);
  extern float atanbuf_[257 * 2];
  extern double datanbuf_[513 * 2];
}

namespace  fastmath {

  // =====================================================================
  // arctan, single-precision; returns phi and r (or 1/r if overR=true)
  // =====================================================================
  inline std::pair<float,float> atan2r(float y_, float x_, bool overR=false) {
    using namespace fastmath_details;
    float mag2 = x_ * x_ + y_ * y_;
    if(!(mag2 > 0))  {  return std::pair<float,float>(0.f,0.f); }   // degenerate case
    
    // float r_ = std::sqrt(mag2);
    float rinv = invSqrt(mag2);
    unsigned int flags = 0;
    float x, y;
    union {
      float f;
      int i;
    } yp;
    yp.f = 32768.f;
    if (y_ < 0 ) { flags |= 4; y_ = -y_; }
    if (x_ < 0 ) { flags |= 2; x_ = -x_; }
    if (y_ > x_) {
      flags |= 1;
      x = rinv * y_; y = rinv * x_; yp.f += y;
    }
    else {
      x = rinv * x_; y = rinv * y_; yp.f += y;
    }
    int ind = (yp.i & 0x01FF) * 2;
    
    float* asbuf = (float*)(atanbuf_ + ind);
    float sv = yp.f - 32768.f;
    float cv = asbuf[0];
    float asv = asbuf[1];
    sv = y * cv - x * sv;    // delta sin value
    // ____ compute arcsin directly
    float asvd = 6.f + sv * sv;   sv *= float(1.0f / 6.0f);
    float th = asv + asvd * sv;
    if (flags & 1) { th = (_2pif / 4.f) - th; }
    if (flags & 2) { th = (_2pif / 2.f) - th; }
    if (flags & 4) { th = -th; }
    return std::pair<float,float>(th,overR ? rinv : rinv*mag2);    
  }
  
  // =====================================================================
  // arctan, double-precision; returns phi and r (or 1/r if overR=true)
  // =====================================================================
  inline std::pair<double, double> atan2r(double y_, double x_, bool overR=false) {
    using namespace fastmath_details;
    // assert(ataninited);
    double mag2 = x_ * x_ + y_ * y_;
    if(!(mag2 > 0)) { return std::pair<double, double>(0.,0.); }   // degenerate case
    
    double r_ = std::sqrt(mag2);
    double rinv = 1./r_;
    unsigned int flags = 0;
    double x, y;
    const double _2p43 = 65536.0 * 65536.0 * 2048.0;
    union {
      double d;
      int i[2];
    } yp;

    yp.d = _2p43;
    if (y_ < 0) { flags |= 4; y_ = -y_; }
    if (x_ < 0) { flags |= 2; x_ = -x_; }
    if (y_ > x_) {
      flags |= 1;
      x = rinv * y_; y = rinv * x_; yp.d += y;
    }
    else {
      x = rinv * x_; y = rinv * y_; yp.d += y;
    }

    int ind = (yp.i[0] & 0x03FF) * 2;  // 0 for little indian
    
    double* dasbuf = (double*)(datanbuf_ + ind);
    double sv = yp.d - _2p43; // index fraction
    double cv = dasbuf[0];
    double asv = dasbuf[1];
    sv = y * cv - x * sv;    // delta sin value
    // double sv = y *(cv-x);
    // ____ compute arcsin directly
    double asvd = 6 + sv * sv;   sv *= double(1.0 / 6.0);
    double th = asv + asvd * sv;
    if (flags & 1) { th = (_2pi / 4) - th; }
    if (flags & 2) { th = (_2pi / 2) - th; }
    if (flags & 4) { th = -th; }
    return std::pair<double,double>(th,overR ? rinv : r_);    
  }
 
  // return eta phi saving some computation
  template<typename T>
  inline  std::pair<T,T> etaphi(T x, T y, T z) {
    std::pair<T,T> por = atan2r(y,x, true);
    x = z*por.second;
    return std::pair<float,float>( std::log(x+std::sqrt(x*x+T(1))), por.first);
  }

}


#endif
