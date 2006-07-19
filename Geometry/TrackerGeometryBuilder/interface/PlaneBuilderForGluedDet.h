#ifndef Geometry_TrackerGeometryBuilder_PlaneBuilderForGluedDet_H
#define Geometry_TrackerGeometryBuilder_PlaneBuilderForGluedDet_H

#include "Geometry/Surface/interface/BoundPlane.h"
#include "Geometry/Surface/interface/ReferenceCounted.h"
#include "Geometry/Surface/interface/RectangularPlaneBounds.h"
#include "Geometry/Surface/interface/TrapezoidalPlaneBounds.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include <utility>
#include <vector>


/** Builds the minimal rectangular box that contains all input GeomDetUnits fully.
 */

class PlaneBuilderForGluedDet {
public:

  /// Warning, remember to assign this pointer to a ReferenceCountingPointer!
  /// Should be changed to return a ReferenceCountingPointer<BoundPlane>
  typedef ReferenceCountingPointer<BoundPlane>  ResultType;

  ResultType plane( const std::vector<const GeomDetUnit*> & dets) const;

  std::pair<RectangularPlaneBounds, GlobalVector> computeRectBounds( const std::vector<const GeomDetUnit*> & dets, const BoundPlane& plane) const;

  Surface::RotationType 
  computeRotation( const std::vector<GeomDetUnit*> & dets, 
		   const Surface::PositionType& meanPos) const; 

};

#endif
