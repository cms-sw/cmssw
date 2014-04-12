#ifndef APPROX_LOG_H
#define APPROX_LOG_H
/*  Quick and dirty, branchless, log implementations
    Author: Florent de Dinechin, Aric, ENS-Lyon 
    All right reserved

Warning + disclaimers:
- no special case handling (infinite/NaN inputs, even zero input, etc)
- no input subnormal handling, you'll get completely wrong results.
  This is the worst problem IMHO (leading to very rare but very bad bugs)
  However it is probable you can guarantee that your input numbers 
  are never subnormal, check that. Otherwise I'll fix it...
- output accuracy reported is only absolute. 
  Relative accuracy may be arbitrary bad around log(1), 
  especially for approx_log0. approx_logf is more or less OK.
- The larger/smaller the input x (i.e. away from 1), the better the accuracy.
- For the higher degree polynomials it is possible to win a few cycles 
  by parallelizing the evaluation of the polynomial (Estrin). 
  It doesn't make much sense if you want to make a vector function. 
- All this code is FMA-safe (and accelerated by FMA)
 
Feel free to distribute or insert in other programs etc, as long as this notice is attached.
    Comments, requests etc: Florent.de.Dinechin@ens-lyon.fr

Polynomials were obtained using Sollya scripts (in comments): 
please also keep these comments attached to the code of approx_logf. 
*/

#include <cstdint>
#include <cmath>
#include <limits>
#include <algorithm>


#ifndef APPROX_MATH_N
#define APPROX_MATH_N
namespace approx_math {
  union binary32 {
    binary32() : ui32(0) {};
    binary32(float ff) : f(ff) {};
    binary32(int32_t ii) : i32(ii){}
    binary32(uint32_t ui) : ui32(ui){}
    
    uint32_t ui32; /* unsigned int */                
    int32_t i32; /* Signed int */                
    float f;
  };
}
#endif


template<int DEGREE>
inline float approx_logf_P(float p);


// the following is Sollya output

// degree =  2   => absolute accuracy is  7 bits
template<>
inline float approx_logf_P<2>(float y) {
  return  y * ( float(0x1.0671c4p0) + y * ( float(-0x7.27744p-4) )) ;
}

// degree =  3   => absolute accuracy is  10 bits
template<>
inline float approx_logf_P<3>(float y) {
  return  y * (float(0x1.013354p0) + y * (-float(0x8.33006p-4) + y * float(0x4.0d16cp-4))) ;
}				  

// degree =  4   => absolute accuracy is  13 bits
template<>
inline float approx_logf_P<4>(float y) {
  return  y * (float(0xf.ff5bap-4) + y * (-float(0x8.13e5ep-4) + y * (float(0x5.826ep-4) + y * (-float(0x2.e87fb8p-4))))) ;
}

// degree =  5   => absolute accuracy is  16 bits
template<>
inline float approx_logf_P<5>(float y) {
  return  y * (float(0xf.ff652p-4) + y * (-float(0x8.0048ap-4) + y * (float(0x5.72782p-4) + y * (-float(0x4.20904p-4) + y * float(0x2.1d7fd8p-4))))) ;
}

// degree =  6   => absolute accuracy is  19 bits
template<>
inline float approx_logf_P<6>(float y) {
  return  y * (float(0xf.fff14p-4) + y * (-float(0x7.ff4bfp-4) + y * (float(0x5.582f6p-4) + y * (-float(0x4.1dcf2p-4) + y * (float(0x3.3863f8p-4) + y * (-float(0x1.9288d4p-4))))))) ;
}

// degree =  7   => absolute accuracy is  21 bits
template<>
inline float approx_logf_P<7>(float y) {
  return  y * (float(0x1.000034p0) + y * (-float(0x7.ffe57p-4) + y * (float(0x5.5422ep-4) + y * (-float(0x4.037a6p-4) + y * (float(0x3.541c88p-4) + y * (-float(0x2.af842p-4) + y * float(0x1.48b3d8p-4))))))) ;
}

// degree =  8   => absolute accuracy is  24 bits
template<>
inline float approx_logf_P<8>(float y) {
   return  y * ( float(0x1.00000cp0) + y * (float(-0x8.0003p-4) + y * (float(0x5.55087p-4) + y * ( float(-0x3.fedcep-4) + y * (float(0x3.3a1dap-4) + y * (float(-0x2.cb55fp-4) + y * (float(0x2.38831p-4) + y * (float(-0xf.e87cap-8) )))))))) ;
}



template<int DEGREE>
inline float unsafe_logf_impl(float x) {
  using namespace approx_math;

  binary32 xx,m;
  xx.f = x;
  
  // as many integer computations as possible, most are 1-cycle only, and lots of ILP.
  int e= (((xx.i32) >> 23) & 0xFF) -127; // extract exponent
  m.i32 = (xx.i32 & 0x007FFFFF) | 0x3F800000; // extract mantissa as an FP number
  
  int adjust = (xx.i32>>22)&1; // first bit of the mantissa, tells us if 1.m > 1.5
  m.i32 -= adjust << 23; // if so, divide 1.m by 2 (exact operation, no rounding)
  e += adjust;           // and update exponent so we still have x=2^E*y
  
  // now back to floating-point
  float y = m.f -1.0f; // Sterbenz-exact; cancels but we don't care about output relative error
  // all the computations so far were free of rounding errors...

  // the following is based on Sollya output
  float p = approx_logf_P<DEGREE>(y);
  

  constexpr float Log2=0xb.17218p-4; // 0.693147182464599609375 
  return float(e)*Log2+p;

}

#ifndef NO_APPROX_MATH
template<int DEGREE>
inline float unsafe_logf(float x) {
  return unsafe_logf_impl<DEGREE>(x);
}

template<int DEGREE>
inline float approx_logf(float x) {
  using namespace approx_math;


  constexpr float MAXNUMF = 3.4028234663852885981170418348451692544e38f;

  //x = std::max(std::min(x,MAXNUMF),0.f);
  float  res = unsafe_logf<DEGREE>(x);
  res =  (x<MAXNUMF) ? res : std::numeric_limits<float>::infinity();
  return (x>0) ? res :std::numeric_limits<float>::quiet_NaN();
}

#else
template<int DEGREE>
inline float unsafe_logf(float x) {
  return std::log(x);
}

template<int DEGREE>
inline float approx_logf(float x) {
  return std::log(x);
}


#endif // NO_APPROX_MATH

#endif
