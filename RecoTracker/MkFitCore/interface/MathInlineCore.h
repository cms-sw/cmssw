#ifndef RecoTracker_MkFitCore_interface_MathInlineCore_h
#define RecoTracker_MkFitCore_interface_MathInlineCore_h

#include <cmath>

namespace mkfit {

  template <typename T>
  inline T sqr(T x) {
    return x * x;
  }
  template <typename T>
  inline T cube(T x) {
    return x * x * x;
  }

  inline float hipo(float x, float y) { return std::sqrt(x * x + y * y); }

  inline float hipo_sqr(float x, float y) { return x * x + y * y; }

}  // namespace mkfit

#endif
