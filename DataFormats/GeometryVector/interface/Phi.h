#ifndef GeometryVector_Geom_Phi_h
#define GeometryVector_Geom_Phi_h

#include "DataFormats/GeometryVector/interface/Pi.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include <cmath>

namespace Geom {

/** \class Phi
 *  A class for azimuthal angle represantation and algebra.
 *  The use of Phi<T> is tranparant due to the implicit conversion to T
 *  Constructs like cos(phi) work as with float or double.
 *  The difference with respect to built-in types is that
 *  Phi is kept in the range (-pi, pi] by default, and this is consistently
 *  implemented in aritmetic operations. In other words, Phi implements 
 *  "modulo(2 pi)" arithmetics.
 *  Phi can be instantiated to implement the range 0 to 2pi.
 */

  using namespace angle0To2pi;

  enum class PhiRange {MinusPiToPi, ZeroTo2pi};

  template <class T1, PhiRange range>
  class NormalizeWrapper {};

  template <class T1>
  class NormalizeWrapper<T1, PhiRange::MinusPiToPi> {
    public:
    static void normalize(T1 &value) { // Reduce range to -pi to pi
      constexpr T1 pi = 1._pi;
      constexpr T1 twoPi = 2._pi;

      if( value > twoPi || value < -twoPi) {
        value = std::fmod( value, twoPi);
      }
      if (value <= -pi) value += twoPi;
      if (value >  pi) value -= twoPi;
    }
  };

  template <class T1>
  class NormalizeWrapper<T1, PhiRange::ZeroTo2pi> { // Reduce range to 0 to 2pi
  public:
    static void normalize(T1 &value) {
      value = make0To2pi(value);
    }
  };

  template <class T1, PhiRange range = PhiRange::MinusPiToPi>
  class Phi {
  public:

    /// Default constructor does not initialise - just as double.
    Phi() {}

    // Constructor from T1.
    // Not "explicit" to enable convenient conversions.
    // There may be cases of ambiguities because of multiple possible
    // conversions, in which case explicit casts must be used.
    // The constructor provides range checking and normalization,
    // e.g. the value of Pi(2*pi()+1) is 1
    Phi( const T1& val) : theValue(val) { normalize(theValue);}

    /// conversion operator makes transparent use possible.
    operator T1() const { return theValue;}

    /// Template argument conversion
    template <class T3, PhiRange range1> operator Phi<T3, range1>() { return Phi<T3, range1>(theValue);}

    /// Explicit access to value in case implicit conversion not OK
    T1 value() const { return theValue;}

    // so that template classes expecting phi() works! (deltaPhi)
    T1 phi() const { return theValue;}

    /// Standard arithmetics 
    Phi& operator+=(const T1& a) {theValue+=a; normalize(theValue); return *this;}
    Phi& operator+=(const Phi& a) {return operator+=(a.value());}

    Phi& operator-=(const T1& a) {theValue-=a; normalize(theValue); return *this;}
    Phi& operator-=(const Phi& a) {return operator-=(a.value());}

    Phi& operator*=(const T1& a) {theValue*=a; normalize(theValue); return *this;}

    Phi& operator/=(const T1& a) {theValue/=a; normalize(theValue); return *this;}

    T1 degrees() const { return convertRadToDeg(theValue); }
    
    // nearZero() tells whether the angle is close enough to 0 to be considered 0.
    // The default tolerance is 1 degree.
    inline bool nearZero(float tolerance = 1.0_deg) const {
      return (std::abs(theValue) - tolerance <= 0.0);
    }

    // nearEqual() tells whether two angles are close enough to be considered equal.
    // The default tolerance is 0.001 radian.
    inline bool nearEqual(const Phi<T1, range> &angle, float tolerance = 0.001) const {
      return (std::abs(theValue - angle) - tolerance <= 0.0);
    }

  private:

    T1 theValue;

    void normalize(T1 &value) {
      NormalizeWrapper<T1, range>::normalize(value);
    }
  };


  /// - operator
  template <class T1, PhiRange range>
  inline Phi<T1, range> operator-(const Phi<T1, range>& a) {return Phi<T1, range>(-a.value());}


  /// Addition
  template <class T1, PhiRange range>
  inline Phi<T1, range> operator+(const Phi<T1, range>& a, const Phi<T1, range>& b) {
    return Phi<T1, range>(a) += b;
  }
  /// Addition with scalar, does not change the precision
  template <class T1, PhiRange range, class Scalar>
  inline Phi<T1, range> operator+(const Phi<T1, range>& a, const Scalar& b) {
    return Phi<T1, range>(a) += b;
  }
  /// Addition with scalar, does not change the precision
  template <class T1, PhiRange range, class Scalar>
  inline Phi<T1, range> operator+(const Scalar& a, const Phi<T1, range>& b) {
    return Phi<T1, range>(b) += a;
  }


  /// Subtraction
  template <class T1, PhiRange range>
  inline Phi<T1, range> operator-(const Phi<T1, range>& a, const Phi<T1, range>& b) { 
    return Phi<T1, range>(a) -= b;
  }
  /// Subtraction with scalar, does not change the precision
  template <class T1, PhiRange range, class Scalar>
  inline Phi<T1, range> operator-(const Phi<T1, range>& a, const Scalar& b) { 
    return Phi<T1, range>(a) -= b;
  }
  /// Subtraction with scalar, does not change the precision
  template <class T1, PhiRange range, class Scalar>
  inline Phi<T1, range> operator-(const Scalar& a, const Phi<T1, range>& b) { 
    return Phi<T1, range>(a - b.value());
  }


  /// Multiplication with scalar, does not change the precision
  template <class T1, PhiRange range, class Scalar>
  inline Phi<T1, range> operator*(const Phi<T1, range>& a, const Scalar& b) {
    return Phi<T1, range>(a) *= b;
  }
  /// Multiplication with scalar
  template <class T1, PhiRange range>
  inline Phi<T1, range> operator*(double a, const Phi<T1, range>& b) {
    return Phi<T1, range>(b) *= a;
  }


  /// Division
  template <class T1, PhiRange range>
  inline T1 operator/(const Phi<T1, range>& a, const Phi<T1, range>& b) { 
    return a.value() / b.value();
  }
  /// Division by scalar
  template <class T1, PhiRange range>
  inline Phi<T1, range> operator/(const Phi<T1, range>& a, double b) {
    return Phi<T1, range>(a) /= b;
  }

  // For convenience
  template<class T>
  using Phi0To2pi = Phi<T, PhiRange::ZeroTo2pi>;
}

/*
// this a full mess wiht the above that is a mess in itself
#include "DataFormats/Math/interface/deltaPhi.h"
namespace reco {
  template <class T1,class T2>
  inline double deltaPhi(const Geom::Phi<T1> phi1, const Geom::Phi<T2> phi2) {
    return deltaPhi(static_cast<double>(phi1.value()), static_cast<double>(phi2.value()));
  }
 
  template <class T>
  inline double deltaPhi(const Geom::Phi<T> phi1, double phi2) {
    return deltaPhi(static_cast<double>(phi1.value()), phi2);
  }
  template <class T>
  inline double deltaPhi(const Geom::Phi<T> phi1, float phi2) {
    return deltaPhi(static_cast<double>(phi1.value()), static_cast<double>(phi2));
  }
  template <class T>
  inline double deltaPhi(double phi1, const Geom::Phi<T>  phi2) {
    return deltaPhi(phi1, static_cast<double>(phi2.value()) );
  }
  template <class T>
  inline double deltaPhi(float phi1, const Geom::Phi<T>  phi2) {
    return deltaPhi(static_cast<double>(phi1),static_cast<double>(phi2.value()) );
  }
}
*/

#endif











