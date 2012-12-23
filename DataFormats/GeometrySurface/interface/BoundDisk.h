#ifndef Geom_BoundDisk_H
#define Geom_BoundDisk_H

#include "DataFormats/GeometrySurface/interface/Plane.h"
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
 *  $Date: 2012/05/05 13:32:09 $
 *  $Revision: 1.4 $
 */

class Disk GCC11_FINAL : public Plane {
public:

  template<typename... Args>
    Disk(Args&& ... args,
	   Scalar radius) :
    Plane(std::forward<Args>(args)...){}
  

  typedef ReferenceCountingPointer<Disk> DiskPointer;
  typedef ConstReferenceCountingPointer<Disk> ConstDiskPointer;
  typedef ReferenceCountingPointer<Disk> BoundDiskPointer;
  typedef ConstReferenceCountingPointer<Disk> ConstBoundDiskPointer;

 template<typename... Args>
  static DiskPointer build(Args&& ... args) {
    return DiskPointer(new Disk(std::forward<Args>(args)...));
  }


  virtual ~Disk() {}


  // -- DEPRECATED CONSTRUCTORS



  // -- Extension of the Surface interface for disk

  /// The inner radius of the disk
  float innerRadius() const { return static_cast<const SimpleDiskBounds&>(bounds()).innerRadius();}

  /// The outer radius of the disk
  float outerRadius() const  { return static_cast<const SimpleDiskBounds&>(bounds()).outerRadius();}


};
using BoundDisk=Disk;

#endif // Geom_BoundDisk_H
