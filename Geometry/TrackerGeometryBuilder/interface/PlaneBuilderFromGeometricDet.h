#ifndef Geometry_TrackerGeometryBuilder_PlaneBuilderFromGeometricDet_H
#define Geometry_TrackerGeometryBuilder_PlaneBuilderFromGeometricDet_H

#include "Geometry/Surface/interface/BoundPlane.h"
#include "Geometry/Surface/interface/ReferenceCounted.h"

class GeometricDet;
/**
 * Converts DDFilteredView volumes to Bounds
 */
class PlaneBuilderFromGeometricDet {
public:
  typedef ReferenceCountingPointer<BoundPlane>  ResultType;
  
  ResultType plane( const GeometricDet*& gd) const;
 private:
  
};

#endif
