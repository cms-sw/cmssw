#include "Geometry/CommonTopologies/interface/StackGeomDet.h"

StackGeomDet::StackGeomDet(BoundPlane* sp,
                           const GeomDetUnit* lowerDet,
                           const GeomDetUnit* upperDet,
                           const DetId stackDetId)
    : TrackerGeomDet(sp), theLowerDet(lowerDet), theUpperDet(upperDet) {
  setDetId(stackDetId);
}

StackGeomDet::~StackGeomDet() {}

std::vector<const GeomDet*> StackGeomDet::components() const {
  return std::vector<const GeomDet*>{theLowerDet, theUpperDet};
}
