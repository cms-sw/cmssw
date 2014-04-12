#ifndef GeometryVector_PV3DBase_h
#define GeometryVector_PV3DBase_h

#include "DataFormats/GeometryVector/interface/Basic3DVector.h"
#include <iosfwd>

/** Base class for Point and Vector classes with restricted set of operations and
 *  coordinate frame compile time checking.
 *  This class restricts the interface of Basic3DVector, removing all algebraic
 *  operations and limiting the constructors. The Point and Vector classes
 *  inherit from this class, and add the relevant algebraic operations.
 */

template <class T, class PVType, class FrameType>
class PV3DBase {
public:

  typedef T                                         ScalarType;
  typedef Basic3DVector<T>                          BasicVectorType;
  typedef typename BasicVectorType::Cylindrical     Cylindrical;
  typedef typename BasicVectorType::Spherical       Spherical;
  typedef typename BasicVectorType::Polar           Polar;
  typedef typename BasicVectorType::MathVector  MathVector;


  /** default constructor uses default constructor of T to initialize the 
   *  components. For built-in floating-point types this means initialization 
   * to zero
   */
  PV3DBase() : theVector() {}

  /// construct from cartesian coordinates
  PV3DBase(const T & x, const T & y, const T & z) : theVector(x, y, z) {}

  /** Construct from cylindrical coordinates.
   */
  PV3DBase( const Cylindrical& set) : theVector( set) {}

  /// construct from polar coordinates
  PV3DBase( const Polar& set) : theVector( set) {}

  /** Deprecated construct from polar coordinates, use 
   *  constructor from Polar( theta, phi, r) instead. 
   */
  PV3DBase( const Geom::Theta<T>& th, 
	    const Geom::Phi<T>& ph, const T& r) : theVector(th,ph,r) {}

  /** Explicit constructor from BasicVectorType, possibly of different precision
   */
  template <class U>
  explicit PV3DBase( const Basic3DVector<U>& v) : theVector(v) {}

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
  T z() const { return basicVector().z();}

  T mag2() const { return basicVector().mag2();}
  T mag() const  { return basicVector().mag();}
  T barePhi() const  { return basicVector().barePhi();}
  Geom::Phi<T> phi() const  { return basicVector().phi();}

  T perp2() const { return basicVector().perp2();}
  T perp() const  { return basicVector().perp();}
  T transverse() const  { return basicVector().transverse();}
  T bareTheta() const { return basicVector().bareTheta();}
  Geom::Theta<T> theta() const { return basicVector().theta();}
  T eta() const   { return basicVector().eta();}

protected:
  BasicVectorType theVector;
};

template <class T, class PV, class F>
inline std::ostream & operator<<( std::ostream& s, const PV3DBase<T,PV,F>& v) {
  return s << v.basicVector();
} 

#endif // GeometryVector_PV3DBase_h
