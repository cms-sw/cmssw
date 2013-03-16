#ifndef GeometryVector_newBasic3DVector_h
#define GeometryVector_newBasic3DVector_h

#include "DataFormats/GeometryVector/interface/Basic2DVector.h"
#include "DataFormats/GeometryVector/interface/Theta.h"
#include "DataFormats/GeometryVector/interface/Phi.h"
#include "DataFormats/GeometryVector/interface/PreciseFloatType.h"
#include "DataFormats/GeometryVector/interface/CoordinateSets.h"
#include "DataFormats/Math/interface/ExtVec.h"
#include <iosfwd>
#include <cmath>

namespace detailsBasic3DVector {
  inline float __attribute__((always_inline)) __attribute__ ((pure))
  eta(float x, float y, float z) { float t(z/std::sqrt(x*x+y*y)); return ::asinhf(t);} 
  inline double __attribute__((always_inline)) __attribute__ ((pure))
  eta(double x, double y, double z) { double t(z/std::sqrt(x*x+y*y)); return ::asinh(t);} 
  inline long double __attribute__((always_inline)) __attribute__ ((pure))
  eta(long double x, long double y, long double z) { long double t(z/std::sqrt(x*x+y*y)); return ::asinhl(t);} 
}


template < typename T> 
class Basic3DVector {
public:

  typedef T                                   ScalarType;
  typedef Vec4<T>                             VectorType;
  typedef Vec4<T>                             MathVector;
  typedef Geom::Cylindrical2Cartesian<T>      Cylindrical;
  typedef Geom::Spherical2Cartesian<T>        Spherical;
  typedef Spherical                           Polar; // synonym
    
  /** default constructor uses default constructor of T to initialize the 
   *  components. For built-in floating-point types this means initialization 
   * to zero??? (force init to 0)
   */
  Basic3DVector() : v{0,0,0,0} {}

  /// Copy constructor from same type. Should not be needed but for gcc bug 12685
  Basic3DVector( const Basic3DVector & p) : 
    v(p.v) {}

  /// Copy constructor and implicit conversion from Basic3DVector of different precision
  template <class U>
  Basic3DVector( const Basic3DVector<U> & p) : 
    v{T(p.v[0]),T(p.v[1]),T(p.v[2]),T(p.v[3])} {}


  /// constructor from 2D vector (X and Y from 2D vector, z set to zero)
  Basic3DVector( const Basic2DVector<T> & p) : 
    v{p.x(),p.y(),0} {}

 
  /** Explicit constructor from other (possibly unrelated) vector classes 
   *  The only constraint on the argument type is that it has methods
   *  x(), y() and z(), and that these methods return a type convertible to T.
   *  Examples of use are
   *   <BR> construction from a Basic3DVector with different precision
   *   <BR> construction from a Hep3Vector
   *   <BR> construction from a coordinate system converter 
   */
  template <class OtherPoint> 
  explicit Basic3DVector( const OtherPoint& p) : 
        v{T(p.x()),T(p.y()),T(p.z())} {}


  // constructor from Vec4
  Basic3DVector(MathVector const& iv) :
  v(iv) {}

  template<class U>
  Basic3DVector(Vec4<U> const& iv) : 
  v{T(iv[0]),T(iv[1]),T(iv[2]),T(iv[3])} {}

  /// construct from cartesian coordinates
  Basic3DVector( const T& x, const T& y, const T& z, const T&w=0) : 
    v{x,y,z,w}{}

  /** Deprecated construct from polar coordinates, use 
   *  <BR> Basic3DVector<T>( Basic3DVector<T>::Polar( theta, phi, r))
   *  instead. 
   */
  template <typename U>
  Basic3DVector( const Geom::Theta<U>& theta, 
		 const Geom::Phi<U>& phi, const T& r) {
    Polar p( theta.value(), phi.value(), r);
    v[0] = p.x(); v[1] = p.y(); v[2] = p.z();
  }

  MathVector const & mathVector() const { return v;}
  MathVector & mathVector() { return v;}

  T operator[](int i) const { return v[i];}
  T & operator[](int i) { return v[i];}


  /// Cartesian x coordinate
  T x() const { return v[0];}

  /// Cartesian y coordinate
  T y() const { return v[1];}

  /// Cartesian z coordinate
  T z() const { return v[2];}

  T w() const { return v[3];}

  Basic2DVector<T> xy() const { return ::xy(v);}

  // equality
  bool operator==(const Basic3DVector& rh) const {
    auto res = v==rh.v;
    return res[0]&res[1]&res[2]&res[3];
  }

  /// The vector magnitude squared. Equivalent to vec.dot(vec)
  T mag2() const { return  ::dot(v,v);}

  /// The vector magnitude. Equivalent to sqrt(vec.mag2())
  T mag() const  { return std::sqrt( mag2());}

  /// Squared magnitude of transverse component 
  T perp2() const { return ::dot2(v,v);}

  /// Magnitude of transverse component 
  T perp() const { return std::sqrt( perp2());}

  /// Another name for perp()
  T transverse() const { return perp();}

  /** Azimuthal angle. The value is returned in radians, in the range (-pi,pi].
   *  Same precision as the system atan2(x,y) function.
   *  The return type is Geom::Phi<T>, see it's documentation.
   */ 
  T barePhi() const {return std::atan2(y(),x());}
  Geom::Phi<T> phi() const {return Geom::Phi<T>(barePhi());}

  /** Polar angle. The value is returned in radians, in the range [0,pi]
   *  Same precision as the system atan2(x,y) function.
   *  The return type is Geom::Phi<T>, see it's documentation.
   */ 
  T bareTheta() const {return std::atan2(perp(),z());}
  Geom::Theta<T> theta() const {return Geom::Theta<T>(std::atan2(perp(),z()));}

  /** Pseudorapidity. 
   *  Does not check for zero transverse component; in this case the behavior 
   *  is as for divide-by zero, i.e. system-dependent.
   */
  // T eta() const { return -log( tan( theta()/2.));} 
  T eta() const { return detailsBasic3DVector::eta(x(),y(),z());} // correct 

  /** Unit vector parallel to this.
   *  If mag() is zero, a zero vector is returned.
   */
  Basic3DVector unit() const {
    T my_mag = mag2();
    return (0!=my_mag) ? (*this)*(T(1)/std::sqrt(my_mag)) : *this;
  }

  /** Operator += with a Basic3DVector of possibly different precision.
   */
  template <class U> 
  Basic3DVector& operator+= ( const Basic3DVector<U>& p) {
    v = v + p.v;
    return *this;
  } 

  /** Operator -= with a Basic3DVector of possibly different precision.
   */
  template <class U> 
  Basic3DVector& operator-= ( const Basic3DVector<U>& p) {
    v = v - p.v;
    return *this;
  } 

  /// Unary minus, returns a vector with components (-x(),-y(),-z())
  Basic3DVector operator-() const { return Basic3DVector(-v);}

  /// Scaling by a scalar value (multiplication)
  Basic3DVector& operator*= ( T t) {
    v = t*v;
    return *this;
  } 

  /// Scaling by a scalar value (division)
  Basic3DVector& operator/= ( T t) {
    //t = T(1)/t;
    v = v/t;
    return *this;
  } 

  /// Scalar product, or "dot" product, with a vector of same type.
  T dot( const Basic3DVector& rh) const { 
    return ::dot(v,rh.v);
  }

  /** Scalar (or dot) product with a vector of different precision.
   *  The product is computed without loss of precision. The type
   *  of the returned scalar is the more precise of the scalar types 
   *  of the two vectors.
   */
  template <class U> 
  typename PreciseFloatType<T,U>::Type dot( const Basic3DVector<U>& lh) const { 
    return Basic3DVector<typename PreciseFloatType<T,U>::Type>(*this)
      .dot(Basic3DVector<typename PreciseFloatType<T,U>::Type>(lh));
  }

  /// Vector product, or "cross" product, with a vector of same type.
  Basic3DVector cross( const Basic3DVector& lh) const {
    return ::cross3(v,lh.v);
  }


  /** Vector (or cross) product with a vector of different precision.
   *  The product is computed without loss of precision. The type
   *  of the returned vector is the more precise of the types 
   *  of the two vectors.   
   */
  template <class U> 
  Basic3DVector<typename PreciseFloatType<T,U>::Type> 
  cross( const Basic3DVector<U>& lh) const {
    return Basic3DVector<typename PreciseFloatType<T,U>::Type>(*this)
      .cross(Basic3DVector<typename PreciseFloatType<T,U>::Type>(lh));
  }

public:
  Vec4<T> v;
}  __attribute__ ((aligned (16)));


namespace geometryDetails {
  std::ostream & print3D(std::ostream& s, double x, double y, double z);
}

/// simple text output to standard streams
template <class T>
inline std::ostream & operator<<( std::ostream& s, const Basic3DVector<T>& v) {
  return geometryDetails::print3D(s, v.x(),v.y(), v.z());
}


/// vector sum and subtraction of vectors of possibly different precision
template <class T>
inline Basic3DVector<T>
operator+( const Basic3DVector<T>& a, const Basic3DVector<T>& b) {
  return a.v+b.v;
}
template <class T>
inline Basic3DVector<T>
operator-( const Basic3DVector<T>& a, const Basic3DVector<T>& b) {
  return a.v-b.v;
}

template <class T, class U>
inline Basic3DVector<typename PreciseFloatType<T,U>::Type>
operator+( const Basic3DVector<T>& a, const Basic3DVector<U>& b) {
  typedef Basic3DVector<typename PreciseFloatType<T,U>::Type> RT;
  return RT(a).v+RT(b).v;
}

template <class T, class U>
inline Basic3DVector<typename PreciseFloatType<T,U>::Type>
operator-( const Basic3DVector<T>& a, const Basic3DVector<U>& b) {
  typedef Basic3DVector<typename PreciseFloatType<T,U>::Type> RT;
  return RT(a).v-RT(b).v;
}

/// scalar product of vectors of same precision
template <class T>
inline T operator*( const Basic3DVector<T>& v1, const Basic3DVector<T>& v2) {
  return v1.dot(v2);
}

/// scalar product of vectors of different precision
template <class T, class U>
inline typename PreciseFloatType<T,U>::Type operator*( const Basic3DVector<T>& v1, 
						       const Basic3DVector<U>& v2) {
  return  v1.dot(v2);
}

/** Multiplication by scalar, does not change the precision of the vector.
 *  The return type is the same as the type of the vector argument.
 */
template <class T>
inline Basic3DVector<T> operator*( const Basic3DVector<T>& v, T t) {
  return v.v*t;
}

/// Same as operator*( Vector, Scalar)
template <class T>
inline Basic3DVector<T> operator*(T t, const Basic3DVector<T>& v) {
  return v.v*t;
}



template <class T, typename S>
inline Basic3DVector<T> operator*(S t,  const Basic3DVector<T>& v) {
  return static_cast<T>(t)*v;
}

template <class T, typename S>
inline Basic3DVector<T> operator*(const Basic3DVector<T>& v, S t) {
  return static_cast<T>(t)*v;
}


/** Division by scalar, does not change the precision of the vector.
 *  The return type is the same as the type of the vector argument.
 */
template <class T>
inline Basic3DVector<T> operator/(const Basic3DVector<T>& v, T t) {
  return v.v/t;
}

template <class T, typename S>
inline Basic3DVector<T> operator/( const Basic3DVector<T>& v, S s) {
  //  T t = S(1)/s; return v*t;
  T t = s;
  return v/t;
}


typedef Basic3DVector<float> Basic3DVectorF;
typedef Basic3DVector<double> Basic3DVectorD;


//  add long double specialization
#include "Basic3DVectorLD.h"

#endif // GeometryVector_Basic3DVector_h


