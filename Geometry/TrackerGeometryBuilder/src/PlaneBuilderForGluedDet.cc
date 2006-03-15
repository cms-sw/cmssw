#include "Geometry/TrackerGeometryBuilder/interface/PlaneBuilderForGluedDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/Surface/interface/RectangularPlaneBounds.h"
#include "Geometry/Surface/interface/TrapezoidalPlaneBounds.h"
#include "Geometry/Surface/interface/Surface.h"
#include "Geometry/CommonDetAlgo/interface/BoundingBox.h"

#include "CLHEP/Units/SystemOfUnits.h"
#include <algorithm>


// Warning, remember to assign this pointer to a ReferenceCountingPointer!
PlaneBuilderForGluedDet::ResultType PlaneBuilderForGluedDet::plane( const std::vector<const GeomDetUnit*>& dets,std::string part) const {
  // find mean position
  typedef Surface::PositionType::BasicVectorType Vector;
  Vector posSum(0,0,0);
  for (std::vector<const GeomDetUnit*>::const_iterator i=dets.begin(); i!=dets.end(); i++) {
    posSum += (**i).surface().position().basicVector();
  }
  Surface::PositionType meanPos( posSum/float(dets.size()));
  
  Surface::RotationType rotation =  dets.front()->surface().rotation();
  //  Surface::RotationType rotation = computeRotation( dets, meanPos);
  BoundPlane tmpPlane( meanPos, rotation);

  if(part=="barrel"){
    std::pair<RectangularPlaneBounds,GlobalVector> bo = computeRectBounds( dets, tmpPlane);
    return new BoundPlane( meanPos+bo.second, 
			   rotation,
			   bo.first);
  }else{
    std::pair<TrapezoidalPlaneBounds,GlobalVector> bo = computeTrapBounds( dets, tmpPlane);
    return new BoundPlane( meanPos+bo.second, 
			   rotation,
			   bo.first);
  }
}

std::pair<TrapezoidalPlaneBounds, GlobalVector> PlaneBuilderForGluedDet::computeTrapBounds( const std::vector<const GeomDetUnit*>& dets, const BoundPlane& plane) const {
  std::vector<GlobalPoint> corners;
  for (std::vector<const GeomDetUnit*>::const_iterator idet=dets.begin();
       idet != dets.end(); idet++) {
    const BoundPlane& plane = dynamic_cast<const BoundPlane&>(dets.front()->surface());
    std::vector<GlobalPoint> dc = BoundingBox().corners(plane);
    corners.insert( corners.end(), dc.begin(), dc.end());
  }
  
  float xmin(0), xmax(0), ymin(0), ymax(0), zmin(0), zmax(0),xlowmin(0),xlowmax(0),xhighmin(0),xhighmax(0),bottomedge(0),topedge(0);
  for (std::vector<GlobalPoint>::const_iterator i=corners.begin();
       i!=corners.end(); i++) {
    LocalPoint p = plane.toLocal(*i);
    if (p.x()<xlowmin&& p.y()<0) xlowmin = p.x();
    if (p.x()>xlowmax&& p.y()<0) xlowmax = p.x();
    if (p.x()<xhighmin&& p.y()>0) xhighmin = p.x();
    if (p.x()>xhighmax&& p.y()>0) xhighmax = p.x();
    if (p.x() < xmin) xmin = p.x();
    if (p.x() > xmax) xmax = p.x();
    if (p.y() < ymin) ymin = p.y();
    if (p.y() > ymax) ymax = p.y();
    if (p.z() < zmin) zmin = p.z();
    if (p.z() > zmax) zmax = p.z();
  }

  LocalVector localOffset( (xmin+xmax)/2., (ymin+ymax)/2., (zmin+zmax)/2.);
  GlobalVector offset( plane.toGlobal(localOffset));
  
  if((xlowmax-xlowmin)/2 < (xhighmax-xhighmin)/2 ){
    bottomedge = (xlowmax-xlowmin)/2; 
    topedge = (xhighmax-xhighmin)/2;
  }else{
    topedge = (xlowmax-xlowmin)/2; 
    bottomedge = (xhighmax-xhighmin)/2;
  }

  std::pair<TrapezoidalPlaneBounds, GlobalVector> result(TrapezoidalPlaneBounds(bottomedge,topedge,(ymax-ymin)/2, (zmax-zmin)/2), offset);

  return result;

}

std::pair<RectangularPlaneBounds, GlobalVector> PlaneBuilderForGluedDet::computeRectBounds( const std::vector<const GeomDetUnit*>& dets, const BoundPlane& plane) const {
  // go over all corners and compute maximum deviations from mean pos.
  std::vector<GlobalPoint> corners;
  for (std::vector<const GeomDetUnit*>::const_iterator idet=dets.begin();
       idet != dets.end(); idet++) {
    const BoundPlane& plane = dynamic_cast<const BoundPlane&>(dets.front()->surface());
    std::vector<GlobalPoint> dc = BoundingBox().corners(plane);
    corners.insert( corners.end(), dc.begin(), dc.end());
  }
  
  float xmin(0), xmax(0), ymin(0), ymax(0), zmin(0), zmax(0);
  for (std::vector<GlobalPoint>::const_iterator i=corners.begin();
       i!=corners.end(); i++) {
    LocalPoint p = plane.toLocal(*i);
    if (p.x() < xmin) xmin = p.x();
    if (p.x() > xmax) xmax = p.x();
    if (p.y() < ymin) ymin = p.y();
    if (p.y() > ymax) ymax = p.y();
    if (p.z() < zmin) zmin = p.z();
    if (p.z() > zmax) zmax = p.z();
  }

  LocalVector localOffset( (xmin+xmax)/2., (ymin+ymax)/2., (zmin+zmax)/2.);
  GlobalVector offset( plane.toGlobal(localOffset));

  std::pair<RectangularPlaneBounds, GlobalVector> result(RectangularPlaneBounds((xmax-xmin)/2, (ymax-ymin)/2, (zmax-zmin)/2), offset);

  return result;
}

Surface::RotationType PlaneBuilderForGluedDet::computeRotation( const std::vector<GeomDetUnit*>& dets, const Surface::PositionType& meanPos) const{

  // choose first mono out-pointing rotation
  // the rotations of GluedDets coincide with the mono part
  // Simply take the x,y of the first Det if z points out,
  // or -x, y if it doesn't
  const BoundPlane& plane = dynamic_cast<const BoundPlane&>(dets.front()->surface());
  //GlobalVector n = plane.normalVector();

  GlobalVector xAxis;
  GlobalVector yAxis;
  GlobalVector planeYAxis = plane.toGlobal( LocalVector( 0, 1, 0));
  if (planeYAxis.z() < 0) yAxis = -planeYAxis;
  else                    yAxis =  planeYAxis;

  GlobalVector planeXAxis = plane.toGlobal( LocalVector( 1, 0, 0));
  GlobalVector n = planeXAxis.cross( planeYAxis);

  if (n.x() * meanPos.x() + n.y() * meanPos.y() > 0) {
    xAxis = planeXAxis;
  }
  else {
    xAxis = -planeXAxis;
  }

  return Surface::RotationType( xAxis, yAxis);
}
