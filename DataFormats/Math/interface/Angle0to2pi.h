#ifndef DataFormats_Math_Angle0to2pi_h
#define DataFormats_Math_Angle0to2pi_h

#include "DataFormats/Math/interface/deltaPhi.h"

namespace angle0to2pi {

  using namespace angle_units::operators;

  // Angle0to2pi class keep its  value >= 0 and < 2pi.

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

    // nearZero() tells whether the angle is close enough to 0 to be considered 0.
    // The default tolerance is 1 degree.
    inline bool nearZero(float tolerance = 1.0_deg) const {
      return (std::abs(ang0to2pi) - tolerance <= 0.0);
    }

    // nearEqual() tells whether two angles are close enough to be considered equal.
    // The default tolerance is 0.001 radian.
    inline bool nearEqual(const Angle0to2pi<valType> &angle, float tolerance = 0.001) const {
      return (std::abs(ang0to2pi - angle) - tolerance <= 0.0);
    }

    // degrees() returns the angle in degrees.
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
