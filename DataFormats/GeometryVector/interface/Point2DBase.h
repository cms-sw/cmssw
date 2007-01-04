#ifndef GeometryVector_Point2DBase_h
#define GeometryVector_Point2DBase_h


#include "DataFormats/GeometryVector/interface/PointTag.h"
#include "DataFormats/GeometryVector/interface/PV2DBase.h"
#include "DataFormats/GeometryVector/interface/Vector2DBase.h"

template <class T, class FrameTag>
class Point2DBase : public PV2DBase< T, PointTag, FrameTag> {
public:

  typedef PV2DBase< T, PointTag, FrameTag>    BaseClass;
  typedef Vector2DBase< T, FrameTag>          VectorType;
  typedef Basic2DVector<T>                    BasicVectorType;
  typedef typename BaseClass::Polar           Polar;

  /** default constructor uses default constructor of T to initialize the 
   *  components. For built-in floating-point types this means initialization 
   * to zero
   */
  Point2DBase() {}

  /** Construct from another point in the same reference frame, possiblly
   *  with different precision
   */ 
  template <class U> 
  Point2DBase( const Point2DBase<U,FrameTag>& p) : BaseClass( p.basicVector()) {}

  /// construct from cartesian coordinates
  Point2DBase(const T& x, const T& y) : BaseClass(x, y) {}

  /// construct from polar coordinates
  explicit Point2DBase( const Polar& set) : BaseClass( set) {}

  /** Explicit constructor from BasicVectorType, bypasses consistency checks
   *  for point/vector and for coordinate frame. To be used as carefully as
   *  e.g. const_cast.
   */
  template <class U>
  explicit Point2DBase( const Basic2DVector<U>& v) : BaseClass(v) {}

  /** A Point can be shifted by a Vector of possibly different precision,
   *  defined in the same coordinate frame
   */
  template <class U> 
  Point2DBase& operator+=( const Vector2DBase< U, FrameTag>& v) {
    this->basicVector() += v.basicVector();
    return *this;
  } 

  template <class U> 
  Point2DBase& operator-=( const Vector2DBase< U, FrameTag>& v) {
    this->basicVector() -= v.basicVector();
    return *this;
  } 

};

/** The sum of a Point and a Vector is a Point. The arguments must be defined
 *  in the same reference frame. The resulting point has the higher precision
 *  of the precisions of the two arguments.
 */
template< typename T, typename U, class Frame>
inline Point2DBase< typename PreciseFloatType<T,U>::Type, Frame> 
operator+( const Point2DBase<T,Frame>& p, const Vector2DBase<U,Frame>& v) {
  typedef Point2DBase< typename PreciseFloatType<T,U>::Type, Frame> RT;
  return RT( p.basicVector() + v.basicVector());
}

/** Same as operator+(Point,Vector) (see above)
 */
template< typename T, typename U, class Frame>
inline Point2DBase< typename PreciseFloatType<T,U>::Type, Frame> 
operator+( const Vector2DBase<T,Frame>& p, const Point2DBase<U,Frame>& v) {
  typedef Point2DBase< typename PreciseFloatType<T,U>::Type, Frame> RT;
  return RT( p.basicVector() + v.basicVector());
}

/** The difference of two points is a vector. The arguments must be defined
 *  in the same reference frame. The resulting vector has the higher precision
 *  of the precisions of the two arguments.
 */
template< typename T, typename U, class Frame>
inline Vector2DBase< typename PreciseFloatType<T,U>::Type, Frame>
operator-( const Point2DBase<T,Frame>& p1, const Point2DBase<U,Frame>& p2) {
  typedef Vector2DBase< typename PreciseFloatType<T,U>::Type, Frame> RT;
  return RT( p1.basicVector() - p2.basicVector());
}

#endif // GeometryVector_Point2DBase_h
