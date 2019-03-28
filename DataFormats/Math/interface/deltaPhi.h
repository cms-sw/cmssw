#ifndef DataFormats_Math_deltaPhi_h
#define DataFormats_Math_deltaPhi_h
/* function to compute deltaPhi
 *
 * Ported from original code in RecoJets 
 * by Fedor Ratnikov, FNAL
 * stabilize range reduction
 */
#include <cmath>

namespace reco {
  
  // reduce to [-pi,pi]
  template<typename T>
  constexpr T  reduceRange(T x) {
   constexpr T o2pi = 1./(2.*M_PI);
   if (std::abs(x) <= T(M_PI)) return x;
   T n = std::round(x*o2pi);
   return x - n*T(2.*M_PI);
  }

  constexpr double deltaPhi(double phi1, double phi2) { 
    return reduceRange(phi1 - phi2);
  }

  constexpr double deltaPhi(float phi1, double phi2) {
    return deltaPhi(static_cast<double>(phi1), phi2);
  }
  
  constexpr double deltaPhi(double phi1, float phi2) {
    return deltaPhi(phi1, static_cast<double>(phi2));
  }
  

  constexpr float deltaPhi(float phi1, float phi2) { 
      return reduceRange(phi1 - phi2);
  }


  template<typename T1, typename T2>
    constexpr auto deltaPhi(T1 const & t1, T2 const & t2)->decltype(deltaPhi(t1.phi(), t2.phi())) {
    return deltaPhi(t1.phi(), t2.phi());
  }      

  template <typename T> 
    constexpr T deltaPhi (T phi1, T phi2) { 
    return reduceRange(phi1 - phi2);
  }
}

// lovely!  VI
using reco::deltaPhi;

template<typename T1, typename T2 = T1>
struct DeltaPhi {
  constexpr
  auto operator()(const T1 & t1, const T2 & t2)->decltype(reco::deltaPhi(t1, t2)) const {
    return reco::deltaPhi(t1, t2);
  }
};


namespace angle_units {
  
  constexpr long double piRadians(M_PIl);  // M_PIl is long double version of pi
  constexpr long double degPerRad = 180. / piRadians; // Degrees per radian
  
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
      return rad * 1.;
    }

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
  }
}

namespace angle0to2pi {

  using namespace angle_units::operators;

  // make0to2pi constrains an angle to be >= 0 and < 2pi.
	// This function is a faster version of reco::reduceRange.
	// In timing tests, it takes about half the time of reco::reduceRange.
	// It also protects against floating-point value drift over repeated calculations.

  template <class valType>
  inline constexpr valType make0to2pi(valType angle) {
    constexpr valType twoPi = 2._pi;
    constexpr valType epsilon = 1.e-13;

    if ((std::abs(angle) <= epsilon) || (std::abs(twoPi - std::abs(angle)) <= epsilon))
      return (0.);
      
    // if statements arranged to promote faster performance
    if (angle < 0.) {
      if (angle >= -twoPi)
        angle += twoPi;
      else angle = fmod(angle, twoPi) + twoPi;
    }
    else if (angle >= twoPi) {
      angle = fmod(angle, twoPi);
    }
    return (angle);
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

    inline bool nearEqual(const Angle0to2pi<valType> &angle, float tolerance = 0.001) const {
      return (std::abs(ang0to2pi - angle) - tolerance <= 0.0);
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
