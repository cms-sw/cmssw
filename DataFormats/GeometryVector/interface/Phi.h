#ifndef GeometryVector_Geom_Phi_h
#define GeometryVector_Geom_Phi_h

#include "DataFormats/GeometryVector/interface/Pi.h"
#include <cmath>

namespace Geom {

/** \class Phi
 *  A class for azimuthal angle represantation and algebra.
 *  The use of Phi<T> is tranparant due to the implicit conversion to T
 *  Constructs like cos(phi) work as with float or double.
 *  The difference with respect to built-in types is that
 *  Phi is kept in the range (-pi, pi], and this is consistently
 *  implemented in aritmetic operations. In other words, Phi implements 
 *  "modulo(2 pi)" arithmetics.
 */

  template <class T>
  class Phi {
  public:

    /// Default constructor does not initialise - just as double.
    Phi() {}

    /// Constructor from T, does not provide automatic conversion.
    /// The constructor provides range checking and normalization,
    /// e.g. the value of Pi(2*pi()+1) is 1
    explicit Phi( const T& val) : theValue(val) { normalize();}

    /// conversion operator makes transparent use possible.
    operator T() const { return theValue;}

    /// Template argument conversion
    template <class T1> operator Phi<T1>() { return Phi<T1>(theValue);}

    /// Explicit access to value in case implicit conversion not OK
    T value() const { return theValue;}

    // so that template classes expecting phi() works! (deltaPhi)
    T phi() const { return theValue;}

    /// Standard arithmetics 
    Phi& operator+=(const T& a) {theValue+=a; normalize(); return *this;}
    Phi& operator+=(const Phi& a) {return operator+=(a.value());}

    Phi& operator-=(const T& a) {theValue-=a; normalize(); return *this;}
    Phi& operator-=(const Phi& a) {return operator-=(a.value());}

    Phi& operator*=(const T& a) {theValue*=a; normalize(); return *this;}

    Phi& operator/=(const T& a) {theValue/=a; normalize(); return *this;}

    T degrees() const { return theValue*180./pi();}

  private:

    T theValue;

    void normalize() { 
      if( theValue > twoPi() || theValue < -twoPi()) {
	theValue = fmod( theValue, (T) twoPi());
      }
      if (theValue <= -pi()) theValue += twoPi();
      if (theValue >  pi()) theValue -= twoPi();
    }

  };

  /// - operator
  template <class T>
  inline Phi<T> operator-(const Phi<T>& a) {return Phi<T>(-a.value());}


  /// Addition
  template <class T>
  inline Phi<T> operator+(const Phi<T>& a, const Phi<T>& b) {
    return Phi<T>(a) += b;
  }
  /// Addition with scalar, does not change the precision
  template <class T, class Scalar>
  inline Phi<T> operator+(const Phi<T>& a, const Scalar& b) {
    return Phi<T>(a) += b;
  }
  /// Addition with scalar, does not change the precision
  template <class T, class Scalar>
  inline Phi<T> operator+(const Scalar& a, const Phi<T>& b) {
    return Phi<T>(b) += a;
  }


  /// Subtraction
  template <class T>
  inline Phi<T> operator-(const Phi<T>& a, const Phi<T>& b) { 
    return Phi<T>(a) -= b;
  }
  /// Subtraction with scalar, does not change the precision
  template <class T, class Scalar>
  inline Phi<T> operator-(const Phi<T>& a, const Scalar& b) { 
    return Phi<T>(a) -= b;
  }
  /// Subtraction with scalar, does not change the precision
  template <class T, class Scalar>
  inline Phi<T> operator-(const Scalar& a, const Phi<T>& b) { 
    return Phi<T>(a - b.value());  // use of unary operators would normalize twice
  }


  /// Multiplication with scalar, does not change the precision
  template <class T, class Scalar>
  inline Phi<T> operator*(const Phi<T>& a, const Scalar& b) {
    return Phi<T>(a) *= b;
  }
  /// Multiplication with scalar
  template <class T>
  inline Phi<T> operator*(double a, const Phi<T>& b) {
    return Phi<T>(b) *= a;
  }


  /// Division
  template <class T>
  inline T operator/(const Phi<T>& a, const Phi<T>& b) { 
    return a.value() / b.value();
  }
  /// Division by scalar
  template <class T>
  inline Phi<T> operator/(const Phi<T>& a, double b) {
    return Phi<T>(a) /= b;
  }


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











