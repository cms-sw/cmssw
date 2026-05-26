#ifndef Geometry_TrackerGeometryBuilder_GluedGeomDet_H
#define Geometry_TrackerGeometryBuilder_GluedGeomDet_H

#include "Geometry/CommonTopologies/interface/DoubleSensGeomDet.h"
#include "DataFormats/DetId/interface/DetId.h"

class GluedGeomDet final : public DoubleSensGeomDet {
public:
  GluedGeomDet(BoundPlane* sp, const GeomDetUnit* monoDet, const GeomDetUnit* stereoDet, DetId gluedDetId)
      : DoubleSensGeomDet(sp, monoDet, stereoDet, gluedDetId) {}

  ~GluedGeomDet() override = default;

  const GeomDetUnit* monoDet() const { return firstDet(); }
  const GeomDetUnit* stereoDet() const { return secondDet(); }
};

#endif
