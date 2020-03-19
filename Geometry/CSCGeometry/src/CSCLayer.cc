
#include <Geometry/CSCGeometry/interface/CSCLayer.h>

GlobalPoint CSCLayer::centerOfStrip(int strip) const {
  float stripX = geometry()->xOfStrip(strip);
  GlobalPoint globalPoint = surface().toGlobal(LocalPoint(stripX, 0., 0.));
  return globalPoint;
}

GlobalPoint CSCLayer::centerOfWireGroup(int wireGroup) const {
  //  float y = yOfWireGroup(wireGroup);
  //  GlobalPoint globalPoint = toGlobal(LocalPoint(0., y, 0.));
  GlobalPoint globalPoint = surface().toGlobal(geometry()->localCenterOfWireGroup(wireGroup));
  return globalPoint;
}
