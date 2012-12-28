#ifndef Geom_Plane_H
#define Geom_Plane_H

/** \class Plane
 *
 *  A plane in 3D space.
 *  
 *  \warning Surfaces are reference counted, so only ReferenceCountingPointer
 *  should be used to point to them. For this reason, they should be 
 *  using the static build() method. 
 *  (The normal constructor will become private in the future).
 */

#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "boost/intrusive_ptr.hpp" 

class Plane : public Surface {
public:
#ifndef CMS_NOCXX11
 template<typename... Args>
  Plane(Args&& ... args) :
    Surface(std::forward<Args>(args)...){computeSpan();}
#endif  

  typedef ReferenceCountingPointer<Plane> PlanePointer;
  typedef ConstReferenceCountingPointer<Plane> ConstPlanePointer;
  typedef ReferenceCountingPointer<Plane> BoundPlanePointer;
  typedef ConstReferenceCountingPointer<Plane> ConstBoundPlanePointer;


#ifndef CMS_NOCXX11
  /// Construct a Plane.
  /// The reference frame is defined by pos and rot; the plane is
  /// orthogonal to the local Z axis.
 template<typename... Args>
  static PlanePointer build(Args&& ... args) {
    return PlanePointer(new Plane(std::forward<Args>(args)...));
  }
#endif
 
  ~Plane(){}

// extension of Surface interface for planes

  GlobalVector normalVector() const {
    return GlobalVector( rotation().zx(), rotation().zy(), rotation().zz());
  }

  /// Fast access to distance from plane for a point.
  float localZ (const GlobalPoint& gp) const {
    return normalVector().dot(gp-position());
  }

  /// Fast access to component perpendicular to plane for a vector.
  float localZ (const GlobalVector& gv) const {
    return normalVector().dot(gv);
  }


  void computeSpan() { if(theBounds) theBounds->computeSpan(*this);}


// implementation of Surface interface    

  virtual SurfaceOrientation::Side side( const LocalPoint& p, Scalar toler) const GCC11_FINAL {
    return (std::abs(p.z())<toler) ? SurfaceOrientation::onSurface : 
	(p.z()>0 ? SurfaceOrientation::positiveSide : SurfaceOrientation::negativeSide);
  }

  virtual SurfaceOrientation::Side side( const GlobalPoint& p, Scalar toler) const GCC11_FINAL {
    Scalar lz = localZ(p);
    return (std::abs(lz)<toler ? SurfaceOrientation::onSurface : 
	    (lz>0 ? SurfaceOrientation::positiveSide : SurfaceOrientation::negativeSide));
  }

  /// tangent plane to surface from global point
  virtual ReferenceCountingPointer<TangentPlane> tangentPlane (const GlobalPoint&) const GCC11_FINAL;

  /// tangent plane to surface from local point
  virtual ReferenceCountingPointer<TangentPlane> tangentPlane (const LocalPoint&) const GCC11_FINAL;

};
#ifndef CMS_NOCXX11
using BoundPlane = Plane;
#endif

#endif
