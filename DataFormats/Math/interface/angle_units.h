#ifndef DataFormats_Math_Angle_Units_h
#define DataFormats_Math_Angle_Units_h

#include <cmath>

namespace angle_units {

  constexpr double piRadians(M_PI);
  constexpr double degPerRad = 180. / piRadians;  // Degrees per radian

  namespace operators {

    // Angle
    constexpr double operator"" _pi(long double x) { return double(x) * M_PI; }
    constexpr double operator"" _pi(unsigned long long int x) { return double(x) * M_PI; }
    constexpr double operator"" _deg(long double deg) { return deg / degPerRad; }
    constexpr double operator"" _deg(unsigned long long int deg) { return deg / degPerRad; }
    constexpr double operator"" _rad(long double rad) { return rad * 1.; }

    template <class NumType>
    inline constexpr NumType convertRadToDeg(NumType radians)  // Radians -> degrees
    {
      return (radians * degPerRad);
    }

    template <class NumType>
    inline constexpr double convertDegToRad(NumType degrees)  // Degrees -> radians
    {
      return (degrees / degPerRad);
    }

    template <class NumType>
    typename std::enable_if<!std::numeric_limits<NumType>::is_integer, bool>::type almostEqual(NumType x,
                                                                                               NumType y,
                                                                                               int ulp) {
      return std::fabs(x - y) <= std::numeric_limits<NumType>::epsilon() * std::fabs(x + y) * ulp ||
             std::fabs(x - y) < std::numeric_limits<NumType>::min();
    }

  }  // namespace operators
}  // namespace angle_units

#endif
