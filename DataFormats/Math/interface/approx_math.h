#ifndef DataFormatsMathApproxMath_H
#define DataFormatsMathApproxMath_H

#include <cstdint>
#include <cmath>
#include <limits>
#include <algorithm>

namespace approx_math {
  // not c++ compliant (only C compliant)
  // to be c++ compliaint one must use memcpy...
  union binary32 {
    constexpr binary32() : ui32(0){};
    constexpr binary32(float ff) : f(ff){};
    constexpr binary32(int32_t ii) : i32(ii) {}
    constexpr binary32(uint32_t ui) : ui32(ui) {}

    uint32_t ui32; /* unsigned int */
    int32_t i32;   /* Signed int */
    float f;
  };
#ifdef __SSE4_1__
  constexpr float fpfloor(float x) { return std::floor(x); }
#else
  constexpr float fpfloor(float x) {
    int32_t ret = x;
    binary32 xx(x);
    ret -= (xx.ui32 >> 31);
    return ret;
  }
#endif
}  // namespace approx_math

#endif
