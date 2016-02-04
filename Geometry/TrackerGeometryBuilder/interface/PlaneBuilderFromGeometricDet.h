#ifndef Geometry_TrackerGeometryBuilder_PlaneBuilderFromGeometricDet_H
#define Geometry_TrackerGeometryBuilder_PlaneBuilderFromGeometricDet_H

#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"

class GeometricDet;
/**
 * Converts DDFilteredView volumes to Bounds
 */
class PlaneBuilderFromGeometricDet {
public:
  typedef ReferenceCountingPointer<BoundPlane>  ResultType;
  
  ResultType plane( const GeometricDet* gd) const;
 private:
  
};

#endif
