#ifndef DataFormats_Math_Angle_Units_h
#define DataFormats_Math_Angle_Units_h

#include <cmath>

namespace angle_units {

  constexpr long double piRadians(M_PIl);              // M_PIl is long double version of pi
  constexpr long double degPerRad = 180. / piRadians;  // Degrees per radian

  namespace operators {

    // Angle
    constexpr long double operator"" _pi(long double x) { return x * piRadians; }
    constexpr long double operator"" _pi(unsigned long long int x) { return x * piRadians; }
    constexpr long double operator"" _deg(long double deg) { return deg / degPerRad; }
    constexpr long double operator"" _deg(unsigned long long int deg) { return deg / degPerRad; }
    constexpr long double operator"" _rad(long double rad) { return rad * 1.; }

    template <class NumType>
    inline constexpr NumType convertRadToDeg(NumType radians)  // Radians -> degrees
    {
      return (radians * degPerRad);
    }

    template <class NumType>
    inline constexpr long double convertDegToRad(NumType degrees)  // Degrees -> radians
    {
      return (degrees / degPerRad);
    }
  }  // namespace operators
}  // namespace angle_units

namespace angle0to2pi {

  using angle_units::operators::operator""_pi;

  // make0To2pi constrains an angle to be >= 0 and < 2pi.
  // This function is a faster version of reco::reduceRange.
  // In timing tests, it is almost always faster than reco::reduceRange.
  // It also protects against floating-point value drift over repeated calculations.
  // This implementation uses multiplication instead of division and avoids
  // calling fmod to improve performance.

  template <class valType>
  inline constexpr valType make0To2pi(valType angle) {
    constexpr valType twoPi = 2._pi;
    constexpr valType oneOverTwoPi = 1. / twoPi;
    constexpr valType epsilon = 1.e-13;

    if ((std::abs(angle) <= epsilon) || (std::abs(twoPi - std::abs(angle)) <= epsilon))
      return (0.);
    if (std::abs(angle) > twoPi) {
      valType nFac = trunc(angle * oneOverTwoPi);
      angle -= (nFac * twoPi);
      if (std::abs(angle) <= epsilon)
        return (0.);
    }
    if (angle < 0.)
      angle += twoPi;
    return (angle);
  }
}  // namespace angle0to2pi

#endif
