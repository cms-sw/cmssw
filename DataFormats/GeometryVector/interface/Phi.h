#ifndef GeometryVector_Geom_Phi_h
#define GeometryVector_Geom_Phi_h

#include "DataFormats/GeometryVector/interface/Pi.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/angle_units.h"
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

  using angle_units::operators::operator""_deg;
  using angle_units::operators::convertRadToDeg;

  struct MinusPiToPi {};  // Dummy struct to indicate -pi to pi range
  struct ZeroTo2pi {};    // Dummy struct to indicate 0 to 2pi range

  template <typename T1, typename Range>
  class NormalizeWrapper {};

  template <typename T1>
  class NormalizeWrapper<T1, MinusPiToPi> {
  public:
    static void normalize(T1& value) {  // Reduce range to -pi to pi
      if (value > twoPi() || value < -twoPi()) {
        value = std::fmod(value, static_cast<T1>(twoPi()));
      }
      if (value <= -pi())
        value += twoPi();
      if (value > pi())
        value -= twoPi();
    }
  };

  template <typename T1>
  class NormalizeWrapper<T1, ZeroTo2pi> {  // Reduce range to 0 to 2pi
  public:
    static void normalize(T1& value) { value = angle0to2pi::make0To2pi(value); }
  };

  template <typename T1, typename Range = MinusPiToPi>
  class Phi {
  public:
    /// Default constructor does not initialise - just as double.
    Phi() {}

    // Constructor from T1.
    // Not "explicit" to enable convenient conversions.
    // There may be cases of ambiguities because of multiple possible
    // conversions, in which case explicit casts must be used.
    // The constructor provides range checking and normalization,
    // e.g. the value of Phi(2*pi()+1) is 1
    Phi(const T1& val) : theValue(val) { normalize(theValue); }

    /// conversion operator makes transparent use possible.
    operator T1() const { return theValue; }

    /// Template argument conversion
    template <typename T2, typename Range1>
    operator Phi<T2, Range1>() {
      return Phi<T2, Range1>(theValue);
    }

    /// Explicit access to value in case implicit conversion not OK
    T1 value() const { return theValue; }

    // so that template classes expecting phi() works! (deltaPhi)
    T1 phi() const { return theValue; }

    /// Standard arithmetics
    Phi& operator+=(const T1& a) {
      theValue += a;
      normalize(theValue);
      return *this;
    }
    Phi& operator+=(const Phi& a) { return operator+=(a.value()); }

    Phi& operator-=(const T1& a) {
      theValue -= a;
      normalize(theValue);
      return *this;
    }
    Phi& operator-=(const Phi& a) { return operator-=(a.value()); }

    Phi& operator*=(const T1& a) {
      theValue *= a;
      normalize(theValue);
      return *this;
    }

    Phi& operator/=(const T1& a) {
      theValue /= a;
      normalize(theValue);
      return *this;
    }

    T1 degrees() const { return convertRadToDeg(theValue); }

    // nearZero() tells whether the angle is close enough to 0 to be considered 0.
    // The default tolerance is 1 degree.
    inline bool nearZero(float tolerance = 1.0_deg) const { return (std::abs(theValue) - tolerance <= 0.0); }

    // nearEqual() tells whether two angles are close enough to be considered equal.
    // The default tolerance is 0.001 radian.
    inline bool nearEqual(const Phi<T1, Range>& angle, float tolerance = 0.001) const {
      return (std::abs(theValue - angle) - tolerance <= 0.0);
    }

  private:
    T1 theValue;

    void normalize(T1& value) { NormalizeWrapper<T1, Range>::normalize(value); }
  };

  /// - operator
  template <typename T1, typename Range>
  inline Phi<T1, Range> operator-(const Phi<T1, Range>& a) {
    return Phi<T1, Range>(-a.value());
  }

  /// Addition
  template <typename T1, typename Range>
  inline Phi<T1, Range> operator+(const Phi<T1, Range>& a, const Phi<T1, Range>& b) {
    return Phi<T1, Range>(a) += b;
  }
  /// Addition with scalar, does not change the precision
  template <typename T1, typename Range, typename Scalar>
  inline Phi<T1, Range> operator+(const Phi<T1, Range>& a, const Scalar& b) {
    return Phi<T1, Range>(a) += b;
  }
  /// Addition with scalar, does not change the precision
  template <typename T1, typename Range, typename Scalar>
  inline Phi<T1, Range> operator+(const Scalar& a, const Phi<T1, Range>& b) {
    return Phi<T1, Range>(b) += a;
  }

  /// Subtraction
  template <typename T1, typename Range>
  inline Phi<T1, Range> operator-(const Phi<T1, Range>& a, const Phi<T1, Range>& b) {
    return Phi<T1, Range>(a) -= b;
  }
  /// Subtraction with scalar, does not change the precision
  template <typename T1, typename Range, typename Scalar>
  inline Phi<T1, Range> operator-(const Phi<T1, Range>& a, const Scalar& b) {
    return Phi<T1, Range>(a) -= b;
  }
  /// Subtraction with scalar, does not change the precision
  template <typename T1, typename Range, typename Scalar>
  inline Phi<T1, Range> operator-(const Scalar& a, const Phi<T1, Range>& b) {
    return Phi<T1, Range>(a - b.value());
  }

  /// Multiplication with scalar, does not change the precision
  template <typename T1, typename Range, typename Scalar>
  inline Phi<T1, Range> operator*(const Phi<T1, Range>& a, const Scalar& b) {
    return Phi<T1, Range>(a) *= b;
  }
  /// Multiplication with scalar
  template <typename T1, typename Range>
  inline Phi<T1, Range> operator*(double a, const Phi<T1, Range>& b) {
    return Phi<T1, Range>(b) *= a;
  }

  /// Division
  template <typename T1, typename Range>
  inline T1 operator/(const Phi<T1, Range>& a, const Phi<T1, Range>& b) {
    return a.value() / b.value();
  }
  /// Division by scalar
  template <typename T1, typename Range>
  inline Phi<T1, Range> operator/(const Phi<T1, Range>& a, double b) {
    return Phi<T1, Range>(a) /= b;
  }

  // For convenience
  template <typename T>
  using Phi0To2pi = Phi<T, ZeroTo2pi>;
}  // namespace Geom

/*
// this a full mess with the above that is a mess in itself
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
