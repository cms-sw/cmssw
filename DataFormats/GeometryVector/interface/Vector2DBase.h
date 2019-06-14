#ifndef GeometryVector_Vector2DBase_h
#define GeometryVector_Vector2DBase_h

#include "DataFormats/GeometryVector/interface/VectorTag.h"
#include "DataFormats/GeometryVector/interface/PV2DBase.h"

template <class T, class FrameTag>
class Vector2DBase : public PV2DBase<T, VectorTag, FrameTag> {
public:
  typedef PV2DBase<T, VectorTag, FrameTag> BaseClass;
  typedef Basic2DVector<T> BasicVectorType;
  typedef typename BaseClass::Polar Polar;

  /** default constructor uses default constructor of T to initialize the 
   *  components. For built-in floating-point types this means initialization 
   * to zero
   */
  Vector2DBase() {}

  /** Construct from another vector in the same reference frame, possiblly
   *  with different precision
   */
  template <class U>
  Vector2DBase(const Vector2DBase<U, FrameTag>& p) : BaseClass(p.basicVector()) {}

  /// construct from cartesian coordinates
  Vector2DBase(const T& x, const T& y) : BaseClass(x, y) {}

  /// construct from polar coordinates
  explicit Vector2DBase(const Polar& set) : BaseClass(set) {}

  /** Explicit constructor from BasicVectorType, bypasses consistency checks
   *  for point/vector and for coordinate frame. To be used as carefully as
   *  e.g. const_cast.
   */
  template <class U>
  explicit Vector2DBase(const Basic2DVector<U>& v) : BaseClass(v) {}

  /** Unit vector parallel to this.
   *  If mag() is zero, a zero vector is returned.
   */
  Vector2DBase unit() const { return Vector2DBase(this->basicVector().unit()); }

  /** Increment by another Vector of possibly different precision,
   *  defined in the same reference frame 
   */
  template <class U>
  Vector2DBase& operator+=(const Vector2DBase<U, FrameTag>& v) {
    this->basicVector() += v.basicVector();
    return *this;
  }

  /** Decrement by another Vector of possibly different precision,
   *  defined in the same reference frame 
   */
  template <class U>
  Vector2DBase& operator-=(const Vector2DBase<U, FrameTag>& v) {
    this->basicVector() -= v.basicVector();
    return *this;
  }

  /// Unary minus, returns a vector with components (-x(),-y())
  Vector2DBase operator-() const { return Vector2DBase(-this->basicVector()); }

  /// Scaling by a scalar value (multiplication)
  Vector2DBase& operator*=(const T& t) {
    this->basicVector() *= t;
    return *this;
  }

  /// Scaling by a scalar value (division)
  Vector2DBase& operator/=(const T& t) {
    this->basicVector() /= t;
    return *this;
  }

  /** Scalar (or dot) product with a vector of possibly different precision,
   *  defined in the same reference frame.
   *  The product is computed without loss of precision. The type
   *  of the returned scalar is the more precise of the scalar types 
   *  of the two vectors.
   */
  template <class U>
  typename PreciseFloatType<T, U>::Type dot(const Vector2DBase<U, FrameTag>& v) const {
    return this->basicVector().dot(v.basicVector());
  }
};

/// vector sum and subtraction of vectors of possibly different precision
template <class T, class U, class FrameTag>
inline Vector2DBase<typename PreciseFloatType<T, U>::Type, FrameTag> operator+(const Vector2DBase<T, FrameTag>& v1,
                                                                               const Vector2DBase<U, FrameTag>& v2) {
  typedef Vector2DBase<typename PreciseFloatType<T, U>::Type, FrameTag> RT;
  return RT(v1.basicVector() + v2.basicVector());
}

template <class T, class U, class FrameTag>
inline Vector2DBase<typename PreciseFloatType<T, U>::Type, FrameTag> operator-(const Vector2DBase<T, FrameTag>& v1,
                                                                               const Vector2DBase<U, FrameTag>& v2) {
  typedef Vector2DBase<typename PreciseFloatType<T, U>::Type, FrameTag> RT;
  return RT(v1.basicVector() - v2.basicVector());
}

/// scalar product of vectors of possibly different precision
template <class T, class U, class FrameTag>
inline typename PreciseFloatType<T, U>::Type operator*(const Vector2DBase<T, FrameTag>& v1,
                                                       const Vector2DBase<U, FrameTag>& v2) {
  return v1.basicVector() * v2.basicVector();
}

/** Multiplication by scalar, does not change the precision of the vector.
 *  The return type is the same as the type of the vector argument.
 */
template <class T, class FrameTag, class Scalar>
inline Vector2DBase<T, FrameTag> operator*(const Vector2DBase<T, FrameTag>& v, const Scalar& s) {
  return Vector2DBase<T, FrameTag>(v.basicVector() * s);
}

/// Same as operator*( Vector, Scalar)
template <class T, class FrameTag, class Scalar>
inline Vector2DBase<T, FrameTag> operator*(const Scalar& s, const Vector2DBase<T, FrameTag>& v) {
  return Vector2DBase<T, FrameTag>(v.basicVector() * s);
}

/** Division by scalar, does not change the precision of the vector.
 *  The return type is the same as the type of the vector argument.
 */
template <class T, class FrameTag, class Scalar>
inline Vector2DBase<T, FrameTag> operator/(const Vector2DBase<T, FrameTag>& v, const Scalar& s) {
  return Vector2DBase<T, FrameTag>(v.basicVector() / s);
}

#endif  // GeometryVector_Vector2DBase_h
