#ifndef DataFormats_Math_Angle0to2pi_h
#define DataFormats_Math_Angle0to2pi_h

#include "DataFormats/Math/interface/GeantUnits.h"


namespace angle0to2pi {

  using namespace geant_units::operators;

  // make0to2pi constrains an angle to be >= 0 and < 2pi.

  template <class valType>
  inline constexpr valType make0to2pi(valType angle) {
    constexpr valType twoPi = 2._pi;
    angle = fmod(angle, twoPi);
    if (angle < 0.) angle += twoPi;
      return angle;
  }


  // Angle0to2pi class keep its  value as >= 0 and < 2pi.
  //
  // nearZero() tells whether the angle is close enough to 0 to be considered 0.
  // The default tolerance is 1 degree.
  //
  // degrees() returns the angle in degrees.

  template <class valType>
  class Angle0to2pi {
  public:

    Angle0to2pi(valType ang) :
    ang0to2pi (make0to2pi(ang))
    {
    }

    inline operator valType() const {
      return (ang0to2pi);
    }

    inline bool nearZero(float tolerance = 1.0_deg) const {
      return (std::abs(ang0to2pi) - tolerance <= 0.0);
    }

    inline valType degrees() const {
      return (convertRadToDeg(ang0to2pi));
    }

    inline Angle0to2pi & operator +=(valType ang) {
      ang0to2pi = make0to2pi(ang0to2pi + ang);
      return (*this);
    }

    inline Angle0to2pi & operator -=(valType ang) {
      ang0to2pi = make0to2pi(ang0to2pi - ang );
      return (*this);
    }

    inline Angle0to2pi & operator *=(valType val) {
      ang0to2pi = make0to2pi(ang0to2pi * val);
      return (*this);
    }

    inline Angle0to2pi & operator /=(valType val) {
      ang0to2pi = make0to2pi(ang0to2pi / val);
      return (*this);
    }

  private:
    valType ang0to2pi;
  };

}

#endif
