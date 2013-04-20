#ifndef DataFormat_Math_SIMDVec_H
#define DataFormat_Math_SIMDVec_H

#if ( defined(IN_DICTBUILD) || defined(__REFLEX__) || defined(__CINT__) ) || (__BIGGEST_ALIGNMENT__<16)
#elif (defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ > 7)) || defined(__clang__)
#define USE_EXTVECT
#elif (defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ > 4)) 
#define USE_SSEVECT
#endif


// to be moved elsewhere
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



#if defined(USE_EXTVECT)  
#include "DataFormats/Math/interface/ExtVec.h"
#elif defined(USE_SSEVECT)
#include "DataFormats/Math/interface/SSEVec.h"
#include "DataFormats/Math/interface/SSERot.h"
#endif

#endif //
