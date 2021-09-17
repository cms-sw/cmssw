#include "Geometry/CommonTopologies/interface/GluedGeomDet.h"

GluedGeomDet::GluedGeomDet(BoundPlane* sp,
                           const GeomDetUnit* monoDet,
                           const GeomDetUnit* stereoDet,
                           const DetId gluedDetId)
    : TrackerGeomDet(sp), theMonoDet(monoDet), theStereoDet(stereoDet) {
  setDetId(gluedDetId);
}

GluedGeomDet::~GluedGeomDet() {}

std::vector<const GeomDet*> GluedGeomDet::components() const {
  return std::vector<const GeomDet*>{theMonoDet, theStereoDet};
}
