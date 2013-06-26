#ifndef VolumeBoundary_H
#define VolumeBoundary_H

/* #include "Utilities/GenUtil/interface/ReferenceCountingPointer.h" */
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
class BoundarySurface;
class BoundVolume;
class Bounds;

/** Base class fro volume boundaries ( == delimiting surfaces).
 *  The VolumeBoundary provides a connection between the volume and 
 *  the surface. It also provides the surface Bounds, which are in 
 *  the reference frame of the surface.
 */

class VolumeBoundary {
public:

  typedef ConstReferenceCountingPointer<BoundarySurface> SurfacePointerType;

  virtual const BoundVolume* volume() const = 0;
  virtual SurfacePointerType surface() const = 0;
  virtual const Bounds* bounds() const = 0;

private:

  
};

#endif
