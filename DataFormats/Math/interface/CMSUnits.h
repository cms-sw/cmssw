#ifndef DataFormats_Math_CMS_Units_h
#define DataFormats_Math_CMS_Units_h

// This file provides units represented with user-defined literals to more easily attach units to numerical values.
// The CMS convention is that centimeter = 1 and GeV = 1

#include "DataFormats/Math/interface/angle_units.h"

namespace cms_units {

  using angle_units::piRadians;  // Needed by files the include this file

  namespace operators {

    // The following are needed by files that include this header
    // Since "using namespace" is prohibited in header files, each
    // name is individually imported with a "using" statement.
    using angle_units::operators::operator""_deg;
    using angle_units::operators::operator""_pi;
    using angle_units::operators::operator""_rad;
    using angle_units::operators::almostEqual;
    using angle_units::operators::convertDegToRad;
    using angle_units::operators::convertRadToDeg;

    // Length
    constexpr double operator"" _mm(long double length) { return length * 0.1; }
    constexpr double operator"" _cm(long double length) { return length * 1.; }
    constexpr double operator"" _m(long double length) { return length * 100.; }
    constexpr double operator"" _cm3(long double length) { return length * 1._cm * 1._cm * 1._cm; }
    constexpr double operator"" _m3(long double length) { return length * 1._m * 1._m * 1._m; }
    constexpr double operator"" _mm(unsigned long long int length) { return length * 0.1; }
    constexpr double operator"" _cm(unsigned long long int length) { return length * 1; }

  }  // namespace operators
}  // namespace cms_units

#endif
