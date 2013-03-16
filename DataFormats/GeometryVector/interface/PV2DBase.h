#ifndef GeometryVector_PV2DBase_h
#define GeometryVector_PV2DBase_h

#include "DataFormats/GeometryVector/interface/Basic2DVector.h"

#include <iosfwd>
#include <ostream>

template <class T, class PVType, class FrameType>
class PV2DBase {
public:

  typedef T                                     ScalarType;
  typedef Basic2DVector<T>                      BasicVectorType;
  typedef typename BasicVectorType::Polar       Polar;
  typedef typename BasicVectorType::MathVector  MathVector;

  /** default constructor uses default constructor of T to initialize the 
   *  components. For built-in floating-point types this means initialization 
   * to zero
   */
  PV2DBase() : theVector() {}

  /// construct from cartesian coordinates
  PV2DBase( const T& x, const T& y) : theVector(x,y) {}

  /// construct from polar coordinates
  PV2DBase( const Polar& set) : theVector( set) {}

  /** Explicit constructor from BasicVectorType, possibly of different precision
   */
  template <class U>
  explicit PV2DBase( const Basic2DVector<U>& v) : theVector(v) {}

  /** Access to the basic vector, use only when the operations on Point and Vector
   *  are too restrictive (preferably never). 
   */
  const BasicVectorType& basicVector() const { return theVector;}
#ifndef __REFLEX__
  MathVector const & mathVector() const { return theVector.v;}
  MathVector & mathVector() { return theVector.v;}
#endif

  
  T x() const { return basicVector().x();}
  T y() const { return basicVector().y();}
  T mag2() const { return basicVector().mag2();}
  T r() const    { return basicVector().r();}
  T mag() const  { return basicVector().mag();}
  T barePhi() const  { return basicVector().barePhi();}
  Geom::Phi<T> phi() const { return basicVector().phi();}

protected:
  // required in the implementation of inherited types...
  BasicVectorType& basicVector() { return theVector;}

  BasicVectorType theVector;

};

template <class T, class PV, class F>
inline std::ostream & operator<<(std::ostream& s, const PV2DBase<T,PV,F>& v) {
  return s << " (" << v.x() << ',' << v.y() << ") ";
} 
#endif // GeometryVector_PV2DBase_h
