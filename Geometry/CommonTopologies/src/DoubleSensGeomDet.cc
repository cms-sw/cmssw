#include "Geometry/CommonTopologies/interface/DoubleSensGeomDet.h"

DoubleSensGeomDet::DoubleSensGeomDet(BoundPlane* sp,
                                     const GeomDetUnit* firstDet,
                                     const GeomDetUnit* secondDet,
                                     const DetId doubleSensDetId)
    : TrackerGeomDet(sp), theFirstDet(firstDet), theSecondDet(secondDet) {
  setDetId(doubleSensDetId);
}

DoubleSensGeomDet::~DoubleSensGeomDet() {}

std::vector<const GeomDet*> DoubleSensGeomDet::components() const {
  return std::vector<const GeomDet*>{theFirstDet, theSecondDet};
}
