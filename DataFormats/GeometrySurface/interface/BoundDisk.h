#ifndef Geom_BoundDisk_H
#define Geom_BoundDisk_H

#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/GeometrySurface/interface/SimpleDiskBounds.h"

/** \class BoundDisk
 *
 *  A BoundDisk is a special BoundPlane that is additionally limited by an
 *  inner and outer radius. 
 * 
 *  \warning Surfaces are reference counted, so only ReferenceCountingPointer
 *  should be used to point to them. For this reason, they should be 
 *  using the static build() method. 
 *  (The normal constructor will become private in the future).
 *
 *  $Date: 2007/01/17 20:58:43 $
 *  $Revision: 1.1 $
 */

class BoundDisk : public BoundPlane {
public:

  typedef ReferenceCountingPointer<BoundDisk> BoundDiskPointer;
  typedef ConstReferenceCountingPointer<BoundDisk> ConstBoundDiskPointer;


  /// Construct a disk with origin at pos and with rotation matrix rot,
  /// with bounds. The bounds you provide are cloned.
  static BoundDiskPointer build(const PositionType& pos, 
				const RotationType& rot, 
				Bounds* bounds,
				MediumProperties* mp=0) {
    return BoundDiskPointer(new BoundDisk(pos, rot, bounds, mp));
  }
  

  /// Construct a disk with origin at pos and with rotation matrix rot,
  /// with bounds. The bounds you provide are cloned.
  static BoundDiskPointer build(const PositionType& pos, 
				const RotationType& rot, 
				Bounds& bounds,
				MediumProperties* mp=0) {
    return BoundDiskPointer(new BoundDisk(pos, rot, &bounds, mp));
  }

  virtual ~BoundDisk() {}


  // -- DEPRECATED CONSTRUCTORS

  /// Do not use this constructor directly; use the static build method,
  /// which returns a ReferenceCountingPointer.
  /// This constructor will soon become private
  BoundDisk(const PositionType& pos, 
	    const RotationType& rot, 
	    Bounds* bounds) :
    Surface(pos,rot), BoundPlane( pos, rot, bounds) {}

  /// Do not use this constructor directly; use the static build method,
  /// which returns a ReferenceCountingPointer.
  /// This constructor will soon become private
  BoundDisk(const PositionType& pos, 
	    const RotationType& rot, 
	    const Bounds& bounds) :
    Surface(pos,rot), BoundPlane( pos, rot, bounds) {}


  // -- Extension of the Surface interface for disk

  /// The inner radius of the disk
  float innerRadius() const { return static_cast<const SimpleDiskBounds&>(bounds()).innerRadius();}

  /// The outer radius of the disk
  float outerRadius() const  { return static_cast<const SimpleDiskBounds&>(bounds()).outerRadius();}

 protected:
  // Private constructor - use build() instead
  BoundDisk(const PositionType& pos, 
	    const RotationType& rot, 
	    Bounds* bounds,
	    MediumProperties* mp=0) :
    Surface(pos, rot, mp), BoundPlane(pos, rot, bounds, mp) {}

};


#endif // Geom_BoundDisk_H
