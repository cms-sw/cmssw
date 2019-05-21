#ifndef DataFormats_Math_GeantUnits_h
#define DataFormats_Math_GeantUnits_h

// This file provides units represented with user-defined literals to more easily attach units to numerical values.
// Units here are based upon Geant conventions: millimeter = 1, MeV = 1.
// The CMS convention is that centimeter = 1 and GeV = 1, so care must be taken with code that converts between
// the two conventions.


#include <cmath>


namespace geant_units {
  
  constexpr long double piRadians(M_PI);
  constexpr long double degPerRad = 180. / piRadians; // Degrees per radian
  constexpr long double joule(6.24150e+12);
  constexpr long double seconds(1.e+9);
  constexpr long double nanoseconds(1.);
  
  namespace operators {

    // Angle
    constexpr long double operator "" _pi( long double x ) 
    { return x * piRadians; }
    constexpr long double operator "" _pi( unsigned long long int x ) 
    { return x * piRadians; }
    constexpr long double operator"" _deg( long double deg )
    {
      return deg / degPerRad;
    }
    constexpr long double operator"" _deg( unsigned long long int deg )
    {
      return deg / degPerRad;
    }
    constexpr long double operator"" _rad( long double rad )
    {
      return rad*1.;
    }
    
    // Length 
    constexpr long double operator"" _mm( long double length )
    {
      return length*1.;
    }
    constexpr long double operator"" _cm( long double length )
    {
      return length*10.;
    }
    constexpr long double operator"" _m( long double length )
    {
      return length*1000.;
    }
    constexpr long double operator"" _cm3( long double length )
    {
      return length*1._cm*1._cm*1._cm;
    }
    constexpr long double operator"" _m3( long double length )
    {
      return length*1._m*1._m*1._m;
    }

    // Time
    constexpr long double operator "" _s( long double x ) 
    { return x * seconds; }
    constexpr long double operator "" _ns( long double x ) 
    { return x * nanoseconds; }

    // Energy
    constexpr long double operator "" _MeV( long double energy ) 
    { return energy * 1.; }
    constexpr long double operator "" _eV( long double energy ) 
    { return energy * 1.e-6_MeV; }
    constexpr long double operator "" _TeV( long double energy ) 
    { return energy * 1.e6_MeV; }
    constexpr long double operator "" _GeV( long double energy ) 
    { return energy * 1000._MeV; }

    // Mass
    constexpr long double operator "" _kg( long double mass ) 
    { return mass * ( 1._eV / 1.602176487e-19 ) * 1._s * 1._s / ( 1._m * 1._m ); }
    constexpr long double operator "" _g( long double mass ) 
    { return mass * 1.e-3_kg; }
    constexpr long double operator "" _mg( long double mass ) 
    { return mass * 1.e-3_g; }
    constexpr long double operator "" _mole( long double mass ) 
    { return mass * 1.; }

    // Material properties
    constexpr long double operator"" _mg_per_cm3( long double density )
    { return density * 1._mg / 1._cm3; }
    constexpr long double operator"" _g_per_cm3( long double density )
    { return density * 1._g / 1._cm3; }
    constexpr long double operator"" _g_per_mole( long double mass )
    { return mass * 1._g / 1._mole; }

    template <class NumType>
    inline constexpr NumType convertRadToDeg(NumType radians) // Radians -> degrees
    {
      return (radians * degPerRad);
    }

    template <class NumType>
    inline constexpr long double convertDegToRad(NumType degrees) // Degrees -> radians
    {
      return (degrees / degPerRad);
    }

    template <class NumType>
    inline constexpr NumType convertMmToCm(NumType millimeters) // Millimeters -> centimeters
    {
      return (millimeters / 10.);
    }

    template <class NumType>
    inline constexpr NumType convertCmToMm(NumType centimeters) // Centimeters -> Milliimeters
    {
      return (centimeters * 10.);
    }

    template <class NumType>
    inline constexpr NumType convertMm3ToM3(NumType mm3) // Cubic millimeters -> cubic meters
    {
      return (mm3 / 1.e9);
    }

    template <class NumType>
    inline constexpr NumType convertUnitsTo(long double desiredUnits, NumType val) // Convert Geant units to desired units
    {
      return (val / desiredUnits);
    }
  }
}
  
#endif
