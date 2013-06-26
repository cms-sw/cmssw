#ifndef GeometryVector_Vector3DBase_h
#define GeometryVector_Vector3DBase_h


#include "DataFormats/GeometryVector/interface/VectorTag.h"
#include "DataFormats/GeometryVector/interface/PV3DBase.h"

template <class T, class FrameTag> 
class Vector3DBase : public PV3DBase<T, VectorTag, FrameTag> {
public:

  typedef PV3DBase<T, VectorTag, FrameTag>    BaseClass;
  typedef Vector3DBase< T, FrameTag>          VectorType;
  typedef typename BaseClass::Cylindrical     Cylindrical;
  typedef typename BaseClass::Spherical       Spherical;
  typedef typename BaseClass::Polar           Polar;
  typedef typename BaseClass::BasicVectorType BasicVectorType;

  /** default constructor uses default constructor of T to initialize the 
   *  components. For built-in floating-point types this means initialization 
   * to zero
   */
  Vector3DBase() {}

  /** Construct from another point in the same reference frame, possiblly
   *  with different precision
   */ 
  template <class U> 
  Vector3DBase( const Vector3DBase<U,FrameTag>& v) : BaseClass( v.basicVector()) {}

  /// construct from cartesian coordinates
  Vector3DBase(const T& x, const T& y, const T& z) : BaseClass(x, y, z) {}

  /** Construct from cylindrical coordinates.
   */
  explicit Vector3DBase( const Cylindrical& set) : BaseClass( set) {}

  /// construct from polar coordinates
  explicit Vector3DBase( const Polar& set) : BaseClass( set) {}

  /** Deprecated construct from polar coordinates, use 
   *  constructor from Polar( theta, phi, r) instead. 
   */
  Vector3DBase(const Geom::Theta<T>& th, 
	       const Geom::Phi<T>& ph, const T& r) : BaseClass(th,ph,r) {}

  /** Explicit constructor from BasicVectorType, bypasses consistency checks
   *  for point/vector and for coordinate frame. To be used as carefully as
   *  e.g. const_cast.
   */
  template <class U>
  explicit Vector3DBase( const Basic3DVector<U>& v) : BaseClass(v) {}

  /** Unit vector parallel to this.
   *  If mag() is zero, a zero vector is returned.
   */
  Vector3DBase unit() const { return Vector3DBase( this->basicVector().unit());}

  // equality
  bool operator==(const Vector3DBase & rh) const {
    return this->basicVector()==rh.basicVector();
  }


  /** Increment by another Vector of possibly different precision,
   *  defined in the same reference frame 
   */
  template <class U> 
  Vector3DBase& operator+= ( const Vector3DBase< U, FrameTag>& v) {
    this->theVector += v.basicVector();
    return *this;
  } 
    
  /** Decrement by another Vector of possibly different precision,
   *  defined in the same reference frame 
   */
  template <class U> 
  Vector3DBase& operator-= ( const Vector3DBase< U, FrameTag>& v) {
    this->theVector -= v.basicVector();
    return *this;
  } 
    
  /// Unary minus, returns a vector with components (-x(),-y(),-z())
  Vector3DBase operator-() const { return Vector3DBase(-this->basicVector());}


  /// Scaling by a scalar value (multiplication)
  Vector3DBase& operator*= ( const T& t) { 
    this->theVector *= t;
    return *this;
  } 

  /// Scaling by a scalar value (division)
  Vector3DBase& operator/= ( const T& t) { 
    this->theVector /= t;
    return *this;
  } 

  /** Scalar (or dot) product with a vector of possibly different precision,
   *  defined in the same reference frame.
   *  The product is computed without loss of precision. The type
   *  of the returned scalar is the more precise of the scalar types 
   *  of the two vectors.
   */
  template <class U> 
  typename PreciseFloatType<T,U>::Type 
  dot( const Vector3DBase< U, FrameTag>& v) const { 
    return this->theVector.dot( v.basicVector());
  }

  /** Vector (or cross) product with a vector of possibly different precision,
   *  defined in the same reference frame.
   *  The product is computed without loss of precision. The precision
   *  of the returned Vector is the higher precision of the scalar types 
   *  of the two vectors.
   */
  template <class U> 
  Vector3DBase< typename PreciseFloatType<T,U>::Type, FrameTag> 
  cross( const  Vector3DBase< U, FrameTag>& v) const {
    typedef Vector3DBase< typename PreciseFloatType<T,U>::Type, FrameTag> RT;
    return RT( this->theVector.cross( v.basicVector()));
  }

};

/// vector sum and subtraction of vectors of possibly different precision
template <class T, class U, class FrameTag>
inline Vector3DBase<typename PreciseFloatType<T,U>::Type, FrameTag>
operator+( const Vector3DBase<T, FrameTag>& v1, const Vector3DBase<U, FrameTag>& v2) {
  typedef Vector3DBase<typename PreciseFloatType<T,U>::Type, FrameTag> RT;
  return RT(v1.basicVector() + v2.basicVector());
}

template <class T, class U, class FrameTag>
inline Vector3DBase<typename PreciseFloatType<T,U>::Type, FrameTag>
operator-( const Vector3DBase<T, FrameTag>& v1, const Vector3DBase<U, FrameTag>& v2) {
  typedef Vector3DBase<typename PreciseFloatType<T,U>::Type, FrameTag> RT;
  return RT(v1.basicVector() - v2.basicVector());
}

/// scalar product of vectors of possibly different precision
template <class T, class U, class FrameTag>
inline typename PreciseFloatType<T,U>::Type
operator*( const Vector3DBase<T, FrameTag>& v1, const Vector3DBase<U, FrameTag>& v2) {
  return v1.basicVector() * v2.basicVector();
}

/** Multiplication by scalar, does not change the precision of the vector.
 *  The return type is the same as the type of the vector argument.
 */
template <class T, class FrameTag, class Scalar>
inline Vector3DBase<T, FrameTag> 
operator*( const Vector3DBase<T, FrameTag>& v, const Scalar& s) {
  return Vector3DBase<T, FrameTag>( v.basicVector() * s);
}

/// Same as operator*( Vector, Scalar)
template <class T, class FrameTag, class Scalar>
inline Vector3DBase<T, FrameTag> 
operator*( const Scalar& s, const Vector3DBase<T, FrameTag>& v) {
  return Vector3DBase<T, FrameTag>( v.basicVector() * s);
}

/** Division by scalar, does not change the precision of the vector.
 *  The return type is the same as the type of the vector argument.
 */
template <class T, class FrameTag, class Scalar>
inline Vector3DBase<T, FrameTag> 
operator/( const Vector3DBase<T, FrameTag>& v, const Scalar& s) {
  return Vector3DBase<T, FrameTag>( v.basicVector() / s);
}

#endif // GeometryVector_Vector3DBase_h
