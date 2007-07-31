#ifndef Geom_BoundPlane_H
#define Geom_BoundPlane_H

/** \class BoundPlane
 *
 *  A plane in 3D space, with bounds.
 *  
 *  \warning Surfaces are reference counted, so only ReferenceCountingPointer
 *  should be used to point to them. For this reason, they should be 
 *  using the static build() method. 
 *  (The normal constructor will become private in the future).
 */

#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/BoundSurface.h"
#include "boost/intrusive_ptr.hpp" 


class BoundPlane : public Plane, public BoundSurface {
public:
  typedef ReferenceCountingPointer<BoundPlane> BoundPlanePointer;
  typedef ConstReferenceCountingPointer<BoundPlane> ConstBoundPlanePointer;

  /// Construct a Plane with origin at pos and with rotation matrix rot,
  /// with bounds. The bounds you provide are cloned.
  static BoundPlanePointer build(const PositionType& pos,
				    const RotationType& rot,
				    const Bounds* bounds,
				    MediumProperties* mp=0) {
    return BoundPlanePointer(new BoundPlane(pos, rot, bounds, mp));
  }
  

  /// Construct a Plane with origin at pos and with rotation matrix rot,
  /// with bounds. The bounds you provide are cloned.
  static BoundPlanePointer build(const PositionType& pos,
				    const RotationType& rot,
				    const Bounds& bounds,
				    MediumProperties* mp=0) {
    return BoundPlanePointer(new BoundPlane(pos, rot, bounds, mp));
  }

  virtual ~BoundPlane() {}

  // --DEPRECATED CONSTRUCTORS

  /// Do not use this constructor directly; use the static build method,
  /// which returns a ReferenceCountingPointer.
  /// This constructor will soon become private
  BoundPlane( const PositionType& pos, 
	      const RotationType& rot, 
	      const Bounds* bounds) :
    Surface( pos,rot), Plane( pos, rot), BoundSurface(pos, rot, bounds) {}

  /// Do not use this constructor directly; use the static build method,
  /// which returns a ReferenceCountingPointer.
  /// This constructor will soon become private
  BoundPlane( const PositionType& pos, 
	      const RotationType& rot, 
	      const Bounds& bounds) :
    Surface( pos,rot), Plane( pos, rot), BoundSurface(pos, rot, bounds) {}

  /// Do not use this constructor directly; use the static build method,
  /// which returns a ReferenceCountingPointer.
  /// This constructor will soon become private
  BoundPlane( const PositionType& pos, 
	      const RotationType& rot, 
	      const Bounds* bounds, 
	      MediumProperties* mp) :
    Surface( pos,rot,mp), Plane( pos, rot, mp), BoundSurface(pos, rot, bounds, mp) {}

  /// Do not use this constructor directly; use the static build method,
  /// which returns a ReferenceCountingPointer.
  /// This constructor will soon become private
  BoundPlane( const PositionType& pos, 
	      const RotationType& rot, 
	      const Bounds& bounds, 
	      MediumProperties* mp) :
    Surface( pos,rot,mp), Plane( pos, rot, mp), BoundSurface(pos, rot, bounds, mp) {}

  /// Bound Plane with unlimited bounds
  /// Do not use this constructor directly; use the static build method,
  /// with OpenBounds(), or use a simple Plane instead.
  /// This constructor will soon become private
  BoundPlane( const PositionType& pos, 
	      const RotationType& rot);


};

#endif // Geom_BoundPlane_H
