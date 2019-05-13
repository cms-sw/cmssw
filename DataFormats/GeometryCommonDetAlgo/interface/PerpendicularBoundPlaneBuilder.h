#ifndef PerpendicularBoundPlaneBuilder_H
#define PerpendicularBoundPlaneBuilder_H

#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"

/** Constructs a plane perpendicular to an axis, and oriented in a special way.
 *  Must be updated for reference counting.
 */

class PerpendicularBoundPlaneBuilder {
public:
  BoundPlane* operator()(const Surface::GlobalPoint& origin, const Surface::GlobalVector& perp) const;
};

#endif
