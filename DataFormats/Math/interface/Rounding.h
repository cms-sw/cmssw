#ifndef DataFormats_Math_Rounding_h
#define DataFormats_Math_Rounding_h

// This file provides utilities for comparing and rounding floating point numbers

#include <cmath>

namespace cms_rounding {

  template <class valType>
  inline constexpr valType roundIfNear0(valType value, double tolerance = 1.e-7) {
    if (std::abs(value) < tolerance)
      return (0.0);
    return (value);
  }

  template <class valType>
  inline constexpr valType roundVecIfNear0(valType value, double tolerance = 1.e-7) {
    auto xVal{roundIfNear0(value.x(), tolerance)};
    auto yVal{roundIfNear0(value.y(), tolerance)};
    auto zVal{roundIfNear0(value.z(), tolerance)};
    return (valType{xVal, yVal, zVal});
  }

}  // namespace cms_rounding

#endif
