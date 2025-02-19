#ifndef GeometryVector_Geom_OnePiRange_h
#define GeometryVector_Geom_OnePiRange_h

#include <DataFormats/GeometryVector/interface/Pi.h>
#include <cmath>

namespace Geom {

/** \class OnePiRange 
 *  A class for polar angle represantation.
 *  The use of OnePiRange<T> is tranparant due to the implicit conversion to T
 *  Constructs like cos(theta) work as with float or double.
 *  The difference with respect to built-in types is that
 *  OnePiRange is kept in the range [0, pi], and this is consistently
 *  implemented in aritmetic operations. In other words, OnePiRange implements 
 *  "modulo(pi)" arithmetics.
 */

  template <class T>
  class OnePiRange {
  public:

    /// Default constructor does not initialise - just as double.
    OnePiRange() {}

    /// Constructor from T, does not provide automatic conversion.
    explicit OnePiRange( const T& val) : theValue(val) { normalize(); }

    /// conversion operator makes transparent use possible.
    operator T() const { return theValue;}

    /// Template argument conversion
    template <class T1> operator OnePiRange<T1>() { return OnePiRange<T1>(theValue);}

    /// Explicit access to value in case implicit conversion not OK
    T value() const { return theValue;}

    // Standard arithmetics 
    OnePiRange& operator+=(const T& a) {theValue+=a; normalize(); return *this;}
    OnePiRange& operator+=(const OnePiRange& a) {return operator+=(a.value());}

    OnePiRange& operator-=(const T& a) {theValue-=a; normalize(); return *this;}
    OnePiRange& operator-=(const OnePiRange& a) {return operator-=(a.value());}

    OnePiRange& operator*=(const T& a) {theValue*=a; normalize(); return *this;}

    OnePiRange& operator/=(const T& a) {theValue/=a; normalize(); return *this;}

    T degrees() const { return theValue*180./pi();}

    /// Return the pseudorapidity.
    // No need to handle 0 or pi; in this case "inf" is returned.
    T eta() const { return -log(tan(theValue/2.)); } 
    

  private:

    T theValue;

    void normalize() {
      if( theValue > (T) pi() || theValue < 0) {
	theValue = fmod( theValue, (T) pi());	
      }
      if (theValue <  0.)   theValue += pi();
    }

  };

  /// - operator
  template <class T>
  inline OnePiRange<T> operator-(const OnePiRange<T>& a) {return OnePiRange<T>(-a.value());}


  /// Addition
  template <class T>
  inline OnePiRange<T> operator+(const OnePiRange<T>& a, const OnePiRange<T>& b) {
    return OnePiRange<T>(a) += b; 
  }
  /// Addition with scalar, does not change the precision
  template <class T, class Scalar>
  inline OnePiRange<T> operator+(const OnePiRange<T>& a, const Scalar& b) {
    return OnePiRange<T>(a) += b;
  }
  /// Addition with scalar, does not change the precision
  template <class T, class Scalar>
  inline OnePiRange<T> operator+(const Scalar& a, const OnePiRange<T>& b) {
    return OnePiRange<T>(b) += a;
  }

  /// Subtraction
  template <class T>
  inline OnePiRange<T> operator-(const OnePiRange<T>& a, const OnePiRange<T>& b) { 
    return OnePiRange<T>(a) -= b;
  }

  /// Subtraction with scalar, does not change the precision
  template <class T, class Scalar>
  inline OnePiRange<T> operator-(const OnePiRange<T>& a, const Scalar& b) { 
    return OnePiRange<T>(a) -= b;
  }

  /// Subtraction with scalar, does not change the precision
  template <class T, class Scalar>
  inline OnePiRange<T> operator-(const Scalar& a, const OnePiRange<T>& b) { 
    return OnePiRange<T>(a - b.value());  // use of unary operators would normalize twice
  }

  /// Multiplication with scalar, does not change the precision
  template <class T, class Scalar>
  inline OnePiRange<T> operator*(const OnePiRange<T>& a, const Scalar& b) {
    return OnePiRange<T>(a) *= b;
  }

  /// Multiplication with scalar
  template <class T>
  inline OnePiRange<T> operator*(double a, const OnePiRange<T>& b) {
    return OnePiRange<T>(b) *= a;
  }

  /// Division
  template <class T>
  inline T operator/(const OnePiRange<T>& a, const OnePiRange<T>& b) { 
    return a.value() / b.value();
  }

  /// Division by scalar
  template <class T>
  inline OnePiRange<T> operator/(const OnePiRange<T>& a, double b) {
    return OnePiRange<T>(a) /= b;
  }

}
#endif
