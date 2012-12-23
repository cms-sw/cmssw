#ifndef Geom_TangentPlane_H
#define Geom_TangentPlane_H

#include "DataFormats/GeometrySurface/interface/Plane.h"

/** Plane tangent to a more general surface (e.g. cylinder).
 *  To be constructed by the "parent" surface.
 */

class TangentPlane GCC11_FINAL : public Plane {
public:
  TangentPlane (const PositionType& pos, 
		const RotationType& rot, 
		const Surface* parent) :
    Plane(pos,rot),
    theParent(parent) {}

  /// access to original surface
  const Surface& parentSurface() {return *theParent;}

private:
  ConstReferenceCountingPointer<Surface> theParent;

};
#endif
