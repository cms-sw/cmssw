#ifndef GeometryVector_Point3DBase_h
#define GeometryVector_Point3DBase_h

#include "DataFormats/GeometryVector/interface/PointTag.h"
#include "DataFormats/GeometryVector/interface/PV3DBase.h"
#include "DataFormats/GeometryVector/interface/Point2DBase.h"
#include "DataFormats/GeometryVector/interface/Vector3DBase.h"

template <class T, class FrameTag>
class Point3DBase : public PV3DBase<T, PointTag, FrameTag> {
public:
  typedef PV3DBase<T, PointTag, FrameTag> BaseClass;
  typedef Vector3DBase<T, FrameTag> VectorType;
  typedef typename BaseClass::Cylindrical Cylindrical;
  typedef typename BaseClass::Spherical Spherical;
  typedef typename BaseClass::Polar Polar;
  typedef typename BaseClass::BasicVectorType BasicVectorType;

  /** default constructor uses default constructor of T to initialize the 
   *  components. For built-in floating-point types this means initialization 
   * to zero
   */
  Point3DBase() {}

  /** Construct from another point in the same reference frame, possiblly
   *  with different precision
   */
  template <class U>
  Point3DBase(const Point3DBase<U, FrameTag>& p) : BaseClass(p.basicVector()) {}

  /// construct from cartesian coordinates
  Point3DBase(const T& x, const T& y, const T& z) : BaseClass(x, y, z) {}

  /** Construct from cylindrical coordinates.
   */
  explicit Point3DBase(const Cylindrical& set) : BaseClass(set) {}

  /// construct from polar coordinates
  explicit Point3DBase(const Polar& set) : BaseClass(set) {}

  /** Deprecated construct from polar coordinates, use 
   *  constructor from Polar( theta, phi, r) instead. 
   */
  Point3DBase(const Geom::Theta<T>& th, const Geom::Phi<T>& ph, const T& r) : BaseClass(th, ph, r) {}

  /** Mimick 2D point. This constructor is convenient for points on a plane,
   *  since the z component for them is zero. 
   */
  Point3DBase(const T& x, const T& y) : BaseClass(x, y, 0) {}
  explicit Point3DBase(Point2DBase<T, FrameTag> p) : BaseClass(p.x(), p.y(), 0) {}

  /** Explicit constructor from BasicVectorType, bypasses consistency checks
   *  for point/vector and for coordinate frame. To be used as carefully as
   *  e.g. const_cast.
   */
  template <class U>
  explicit Point3DBase(const Basic3DVector<U>& v) : BaseClass(v) {}

  // equality
  bool operator==(const Point3DBase& rh) const { return this->basicVector() == rh.basicVector(); }

  /** A Point can be shifted by a Vector of possibly different precision,
   *  defined in the same coordinate frame
   */
  template <class U>
  Point3DBase& operator+=(const Vector3DBase<U, FrameTag>& v) {
    this->theVector += v.basicVector();
    return *this;
  }

  template <class U>
  Point3DBase& operator-=(const Vector3DBase<U, FrameTag>& v) {
    this->theVector -= v.basicVector();
    return *this;
  }
};

/** The sum of a Point and a Vector is a Point. The arguments must be defined
 *  in the same reference frame. The resulting point has the higher precision
 *  of the precisions of the two arguments.
 */
template <typename T, typename U, class Frame>
inline Point3DBase<typename PreciseFloatType<T, U>::Type, Frame> operator+(const Point3DBase<T, Frame>& p,
                                                                           const Vector3DBase<U, Frame>& v) {
  typedef Point3DBase<typename PreciseFloatType<T, U>::Type, Frame> RT;
  return RT(p.basicVector() + v.basicVector());
}

/** Same as operator+(Point,Vector) (see above)
 */
template <typename T, typename U, class Frame>
inline Point3DBase<typename PreciseFloatType<T, U>::Type, Frame> operator+(const Vector3DBase<T, Frame>& p,
                                                                           const Point3DBase<U, Frame>& v) {
  typedef Point3DBase<typename PreciseFloatType<T, U>::Type, Frame> RT;
  return RT(p.basicVector() + v.basicVector());
}

/** The difference of two points is a vector. The arguments must be defined
 *  in the same reference frame. The resulting vector has the higher precision
 *  of the precisions of the two arguments.
 */
template <typename T, typename U, class Frame>
inline Vector3DBase<typename PreciseFloatType<T, U>::Type, Frame> operator-(const Point3DBase<T, Frame>& p1,
                                                                            const Point3DBase<U, Frame>& p2) {
  typedef Vector3DBase<typename PreciseFloatType<T, U>::Type, Frame> RT;
  return RT(p1.basicVector() - p2.basicVector());
}

/** The difference of a Point and a Vector is a Point. The arguments must be defined
 *  in the same reference frame. The resulting point has the higher precision
 *  of the precisions of the two arguments.
 */
template <typename T, typename U, class Frame>
inline Point3DBase<typename PreciseFloatType<T, U>::Type, Frame> operator-(const Point3DBase<T, Frame>& p,
                                                                           const Vector3DBase<U, Frame>& v) {
  typedef Point3DBase<typename PreciseFloatType<T, U>::Type, Frame> RT;
  return RT(p.basicVector() - v.basicVector());
}
#endif  // GeometryVector_Point3DBase_h
