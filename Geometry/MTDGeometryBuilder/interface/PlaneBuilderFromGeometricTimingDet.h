#ifndef Geometry_MTDGeometryBuilder_PlaneBuilderFromGeometricTimingDet_H
#define Geometry_MTDGeometryBuilder_PlaneBuilderFromGeometricTimingDet_H

#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"

class GeometricTimingDet;
/**
 * Converts DDFilteredView volumes to Bounds
 */
class PlaneBuilderFromGeometricTimingDet {
public:
  using ResultType = ReferenceCountingPointer<BoundPlane>;
  
  ResultType plane( const GeometricTimingDet* gd) const;
  
};

#endif
