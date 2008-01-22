
#include "Geometry/TrackerGeometryBuilder/interface/PlaneBuilderFromGeometricDet.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

#include <algorithm>

//#define DEBUG

/**
   given a current detector node in the DDFilteredView,
   extract the global translation and rotation.
   Further apply ORCA semantics for the local reference frame in which each
   solid of a detector is defined, in order to get the 'correct' GlobalToLocal
   transforms.
   Further determine the boundaries of the current detector.
  
   TODO:
   . The function should be part of a class.
   . The function currently only knows how to handle BarrelPixel detectors - 
   should also know about other det-types. Maybe several classes, one per
   detector element?  
*/

PlaneBuilderFromGeometricDet::ResultType PlaneBuilderFromGeometricDet::plane(const GeometricDet* gd) const{
  return ResultType( new BoundPlane( gd->positionBounds(), gd->rotationBounds(),gd-> bounds())); 
}	      



