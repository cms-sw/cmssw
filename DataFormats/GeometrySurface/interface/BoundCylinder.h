#ifndef Geom_BoundCylinder_H
#define Geom_BoundCylinder_H

/** \class BoundCylinder
 *
 *  A Cylinder with Bounds.
 *
 *  \warning Surfaces are reference counted, so only ReferenceCountingPointer
 *  should be used to point to them. For this reason, they should be 
 *  using the static build() method. 
 *  (The normal constructor will become private in the future).
 *
 *  $Date: 2007/10/06 20:21:23 $
 *  $Revision: 1.3 $
 */

#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/BoundSurface.h"
#include "boost/intrusive_ptr.hpp" 

class BoundCylinder GCC11_FINAL : public Cylinder, public BoundSurface {
public:

  typedef ReferenceCountingPointer<BoundCylinder>       BoundCylinderPointer;
  typedef ConstReferenceCountingPointer<BoundCylinder>  ConstBoundCylinderPointer;

  /// Construct a cylinder with origin at pos and with rotation matrix rot,
  /// with bounds. The bounds you provide are cloned.
  static BoundCylinderPointer build(const PositionType& pos,
				    const RotationType& rot,
				    Scalar radius,
				    const Bounds* bounds,
				    MediumProperties* mp=0) {
    return BoundCylinderPointer(new BoundCylinder(pos, rot, radius, bounds, mp));
  }
  

  /// Construct a cylinder with origin at pos and with rotation matrix rot,
  /// with bounds. The bounds you provide are cloned.
  static BoundCylinderPointer build(const PositionType& pos,
				    const RotationType& rot,
				    Scalar radius,
				    const Bounds& bounds,
				    MediumProperties* mp=0) {
    return BoundCylinderPointer(new BoundCylinder(pos, rot, radius, &bounds, mp));
  }

  virtual ~BoundCylinder() {}

  // -- DEPRECATED CONSTRUCTORS

  /// Do not use this constructor directly; use the static build method,
  /// which returns a ReferenceCountingPointer.
  /// This constructor will soon become private
  BoundCylinder(const PositionType& pos,
		const RotationType& rot,
		Scalar radius, 
		const Bounds& bounds) :
    Surface( pos,rot ),
    Cylinder( pos, rot, radius), 
    BoundSurface(pos, rot, bounds) 
  { }

  /// Do not use this constructor directly; use the static build method,
  /// which returns a ReferenceCountingPointer.
  /// This constructor will soon become private
  BoundCylinder(const PositionType& pos,
		const RotationType& rot,
		Scalar radius,
		MediumProperties* mp,
		const Bounds& bounds) : 
    Surface( pos,rot ),
    Cylinder( pos, rot, radius, mp ), 
    BoundSurface( pos, rot, bounds ) 
  { }

  /// Obsolete constructor, radius should be given explicitly
  BoundCylinder(const PositionType& pos, 
		const RotationType& rot, 
		const Bounds* bounds);

  /// Obsolete constructor, radius should be given explicitly
  BoundCylinder(const PositionType& pos, 
		const RotationType& rot, 
		const Bounds& bounds);

protected:
  // Private constructor - use build() instead
  BoundCylinder(const PositionType& pos,
		const RotationType& rot,
		Scalar radius,
		const Bounds* bounds,
		MediumProperties* mp=0) : 
    Surface( pos,rot ),
    Cylinder(pos, rot, radius, mp), 
    BoundSurface(pos, rot, bounds, mp)
  { }

};

#endif // Geom_BoundCylinder_H
