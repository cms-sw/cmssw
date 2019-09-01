//#include "Utilities/Configuration/interface/Architecture.h"

#include "DataFormats/GeometryCommonDetAlgo/interface/TransverseBoundPlaneFactory.h"

BoundPlane* TransverseBoundPlaneFactory::operator()(const Surface::GlobalPoint& origin,
                                                    const Surface::GlobalVector& dir) const {
  // z axis perpendicular to transverse momentum
  GlobalVector zAxis = GlobalVector(dir.x(), dir.y(), 0.).unit();

  // y axis coincides with global z
  GlobalVector yAxis(0., 0., 1.);

  // x axis obtained by cross product
  GlobalVector xAxis = (yAxis.cross(zAxis)).unit();

  Surface::RotationType rot(
      xAxis.x(), xAxis.y(), xAxis.z(), yAxis.x(), yAxis.y(), yAxis.z(), zAxis.x(), zAxis.y(), zAxis.z());

  //  Surface::RotationType rot(yAxis, zAxis);

  return new BoundPlane(origin, rot);
}
