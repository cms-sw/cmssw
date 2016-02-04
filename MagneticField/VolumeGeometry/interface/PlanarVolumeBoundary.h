#ifndef PlanarVolumeBoundary_H
#define PlanarVolumeBoundary_H

#include "MagneticField/MagVolumeGeometry/interface/BoundaryPlane.h"

class PlanarVolumeBoundary : public VolumeBoundary {
public:

  typedef ConstReferenceCountingPointer<BoundaryPlane> PlanePointerType;

  PlanarVolumeBoundary( const BoundVolume* vol, PlanePointerType plane, 
			const Bounds* bounds);

  virtual const BoundVolume* volume() const {return theVolume;}
  virtual SurfacePointerType surface() const {return SurfacePointerType(thePlane);}
  virtual const Bounds* bounds() const {return theBounds;}

  PlanePointerType concreteSurface() const {return thePlane;}

private:

  const BoundVolume*        theVolume;
  const PlanePointerType    thePlane;
  const Bounds*             theBounds;

};

#endif
