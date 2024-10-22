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

    // Energy
    constexpr double operator"" _GeV(long double energy) { return energy * 1.; }
    constexpr double operator"" _eV(long double energy) { return energy * 1.e-9_GeV; }
    constexpr double operator"" _MeV(long double energy) { return energy * 1.e-3_GeV; }
    constexpr double operator"" _TeV(long double energy) { return energy * 1.e3_GeV; }

    // Add these conversion functions to this namespace for convenience
    using angle_units::operators::convertCm2ToMm2;
    using angle_units::operators::convertCmToMm;
    using angle_units::operators::convertGeVToKeV;
    using angle_units::operators::convertGeVToMeV;
    using angle_units::operators::convertMeVToGeV;
    using angle_units::operators::convertMm3ToM3;
    using angle_units::operators::convertMmToCm;

  }  // namespace operators
}  // namespace cms_units

#endif
