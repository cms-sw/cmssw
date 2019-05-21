//#include "Utilities/Configuration/interface/Architecture.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/PerpendicularBoundPlaneBuilder.h"

BoundPlane* PerpendicularBoundPlaneBuilder::operator()(const Surface::GlobalPoint& origin,
                                                       const Surface::GlobalVector& perp) const {
  // z axis coincides with perp
  GlobalVector zAxis = perp.unit();

  // x axis has no global Z component
  GlobalVector xAxis;
  if (zAxis.x() != 0 || zAxis.y() != 0) {
    // precision is not an issue here, just protect against divizion by zero
    xAxis = GlobalVector(-zAxis.y(), zAxis.x(), 0).unit();
  } else {  // perp coincides with global Z
    xAxis = GlobalVector(1, 0, 0);
  }

  // y axis obtained by cross product
  GlobalVector yAxis(zAxis.cross(xAxis));

  Surface::RotationType rot(
      xAxis.x(), xAxis.y(), xAxis.z(), yAxis.x(), yAxis.y(), yAxis.z(), zAxis.x(), zAxis.y(), zAxis.z());

  return new BoundPlane(origin, rot);
}
