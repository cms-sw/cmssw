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

class Plane : public virtual Surface {
public:
  typedef ReferenceCountingPointer<Plane> PlanePointer;
  typedef ConstReferenceCountingPointer<Plane> ConstPlanePointer;

  /// Construct a Plane.
  /// The reference frame is defined by pos and rot; the plane is
  /// orthogonal to the local Z axis.
  static PlanePointer build(const PositionType& pos,
				    const RotationType& rot,
				    MediumProperties* mp=0) {
    return PlanePointer(new Plane(pos, rot, mp));
  }

  // -- DEPRECATED CONSTRUCTORS

  /// Do not use this constructor directly; use the static build method,
  /// which returns a ReferenceCountingPointer.
  /// This constructor will soon become private
  Plane( const PositionType& pos, const RotationType& rot) :
    Surface( pos, rot) {}

  /// Do not use this constructor directly; use the static build method,
  /// which returns a ReferenceCountingPointer.
  /// This constructor will soon become private
  Plane( const PositionType& pos, const RotationType& rot, MediumProperties* mp) : 
    Surface( pos, rot, mp) {}

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

#endif
