#ifndef GeometryVector_oldBasic2DVector_h
#define GeometryVector_oldBasic2DVector_h
#if ( defined(IN_DICTBUILD) || defined(__CINT__) )  && !defined(__REFLEX__)
#define __REFLEX__
#endif

#include "DataFormats/GeometryVector/interface/Phi.h"
#include "DataFormats/GeometryVector/interface/PreciseFloatType.h"
#include "DataFormats/GeometryVector/interface/CoordinateSets.h"
#ifndef __REFLEX__ 
#include "DataFormats/Math/interface/SIMDVec.h"
#endif


#include <cmath>
#include <iosfwd>


template < class T> 
class Basic2DVector {
public:
  typedef  Basic2DVector<T> MathVector;

  typedef T    ScalarType;
  typedef Geom::Polar2Cartesian<T>        Polar;

  /** default constructor uses default constructor of T to initialize the 
   *  components. For built-in floating-point types this means initialization 
   * to zero
   */
  Basic2DVector() : theX(0), theY(0) {}

  /// Copy constructor from same type. Should not be needed but for gcc bug 12685
  Basic2DVector( const Basic2DVector & p) : 
    theX(p.x()), theY(p.y()) {}

  /** Explicit constructor from other (possibly unrelated) vector classes 
   *  The only constraint on the argument type is that it has methods
   *  x() and y(), and that these methods return a type convertible to T.
   *  Examples of use are
   *   <BR> construction from a Basic2DVector with different precision
   *   <BR> construction from a coordinate system converter 
   */
  template <class Other> 
  explicit Basic2DVector( const Other& p) : theX(p.x()), theY(p.y()) {}

  /// construct from cartesian coordinates
  Basic2DVector( const T& x, const T& y) : theX(x), theY(y) {}


#if  defined(USE_EXTVECT)
  // constructor from Vec2 or vec4
  template<typename U>
  Basic2DVector(Vec2<U> const& iv) :
    theX(iv[0]), theY(iv[1]) {}
  template<typename U>
  Basic2DVector(Vec4<U> const& iv) :
    theX(iv.arr[0]), theY(iv.arr[1]) {}
#elif  defined(USE_SSEVECT)
  // constructor from Vec2 or vec4
  template<typename U>
  Basic2DVector(mathSSE::Vec2<U> const& iv) :
    theX(iv.arr[0]), theY(iv.arr[1]) {}
  template<typename U>
  Basic2DVector(mathSSE::Vec4<U> const& iv) :
    theX(iv.arr[0]), theY(iv.arr[1]) {}
#endif  

  T operator[](int i) const { return i==0 ?  theX : theY ;}
  T & operator[](int i) { return i==0 ?  theX : theY ;}

  /// Cartesian x coordinate
  T x() const { return theX;}

  /// Cartesian y coordinate
  T y() const { return theY;}

  /// The vector magnitude squared. Equivalent to vec.dot(vec)
  T mag2() const { return theX*theX + theY*theY;}

  /// The vector magnitude. Equivalent to sqrt(vec.mag2())
  T mag() const  { return std::sqrt( mag2());}

  /// Radius, same as mag()
  T r() const    { return mag();}

  /** Azimuthal angle. The value is returned in radians, in the range (-pi,pi].
   *  Same precision as the system atan2(x,y) function.
   *  The return type is Geom::Phi<T>, see it's documentation.
   */ 
  T barePhi() const {return std::atan2(theY,theX);}
  Geom::Phi<T> phi() const {return Geom::Phi<T>(atan2(theY,theX));}

  /** Unit vector parallel to this.
   *  If mag() is zero, a zero vector is returned.
   */
  Basic2DVector unit() const {
    T my_mag = mag();
    return my_mag == 0 ? *this : *this / my_mag;
  }

  /** Operator += with a Basic2DVector of possibly different precision.
   */
  template <class U> 
  Basic2DVector& operator+= ( const Basic2DVector<U>& p) {
    theX += p.x();
    theY += p.y();
    return *this;
  } 

  /** Operator -= with a Basic2DVector of possibly different precision.
   */
  template <class U> 
  Basic2DVector& operator-= ( const Basic2DVector<U>& p) {
    theX -= p.x();
    theY -= p.y();
    return *this;
  } 

  /// Unary minus, returns a vector with components (-x(),-y(),-z())
  Basic2DVector operator-() const { return Basic2DVector(-x(),-y());}

  /// Scaling by a scalar value (multiplication)
  Basic2DVector& operator*= ( const T& t) {
    theX *= t;
    theY *= t;
    return *this;
  } 

  /// Scaling by a scalar value (division)
  Basic2DVector& operator/= ( const T& t) {
    theX /= t;
    theY /= t;
    return *this;
  } 

  /// Scalar product, or "dot" product, with a vector of same type.
  T dot( const Basic2DVector& v) const { return x()*v.x() + y()*v.y();}

  /** Scalar (or dot) product with a vector of different precision.
   *  The product is computed without loss of precision. The type
   *  of the returned scalar is the more precise of the scalar types 
   *  of the two vectors.
   */
  template <class U> 
  typename PreciseFloatType<T,U>::Type dot( const Basic2DVector<U>& v) const { 
    return x()*v.x() + y()*v.y();
  }



  /// Vector product, or "cross" product, with a vector of same type.
  T cross( const Basic2DVector& v) const { return x()*v.y() - y()*v.x();}

  /** Vector (or cross) product with a vector of different precision.
   *  The product is computed without loss of precision. The type
   *  of the returned scalar is the more precise of the scalar types 
   *  of the two vectors.
   */
  template <class U> 
  typename PreciseFloatType<T,U>::Type cross( const Basic2DVector<U>& v) const { 
    return x()*v.y() - y()*v.x();
  }


private:
  T theX;
  T theY;
};


namespace geometryDetails {
  std::ostream & print2D(std::ostream& s, double x, double y);

}

/// simple text output to standard streams
template <class T>
inline std::ostream & operator<<( std::ostream& s, const Basic2DVector<T>& v) {
  return geometryDetails::print2D(s, v.x(),v.y());
}


/// vector sum and subtraction of vectors of possibly different precision
template <class T, class U>
inline Basic2DVector<typename PreciseFloatType<T,U>::Type>
operator+( const Basic2DVector<T>& a, const Basic2DVector<U>& b) {
  typedef Basic2DVector<typename PreciseFloatType<T,U>::Type> RT;
  return RT(a.x()+b.x(), a.y()+b.y());
}

template <class T, class U>
inline Basic2DVector<typename PreciseFloatType<T,U>::Type>
operator-( const Basic2DVector<T>& a, const Basic2DVector<U>& b) {
  typedef Basic2DVector<typename PreciseFloatType<T,U>::Type> RT;
  return RT(a.x()-b.x(), a.y()-b.y());
}




// scalar product of vectors of same precision
template <class T>
inline T operator*( const Basic2DVector<T>& v1, const Basic2DVector<T>& v2) {
  return v1.dot(v2);
}

/// scalar product of vectors of different precision
template <class T, class U>
inline typename PreciseFloatType<T,U>::Type operator*( const Basic2DVector<T>& v1, 
						       const Basic2DVector<U>& v2) {
  return v1.x()*v2.x() + v1.y()*v2.y();
}

/** Multiplication by scalar, does not change the precision of the vector.
 *  The return type is the same as the type of the vector argument.
 */
template <class T, class Scalar>
inline Basic2DVector<T> operator*( const Basic2DVector<T>& v, const Scalar& s) {
  T t = static_cast<T>(s);
  return Basic2DVector<T>(v.x()*t, v.y()*t);
}

/// Same as operator*( Vector, Scalar)
template <class T, class Scalar>
inline Basic2DVector<T> operator*( const Scalar& s, const Basic2DVector<T>& v) {
  T t = static_cast<T>(s);
  return Basic2DVector<T>(v.x()*t, v.y()*t);
}

/** Division by scalar, does not change the precision of the vector.
 *  The return type is the same as the type of the vector argument.
 */
template <class T, class Scalar>
inline Basic2DVector<T> operator/( const Basic2DVector<T>& v, const Scalar& s) {
  T t = static_cast<T>(s);
  return Basic2DVector<T>(v.x()/t, v.y()/t);
}


typedef Basic2DVector<float> Basic2DVectorF;
typedef Basic2DVector<double> Basic2DVectorD;

#endif // GeometryVector_Basic2DVector_h
