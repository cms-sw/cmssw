#ifndef Geometry_CommonTopologies_StackGeomDet_H
#define Geometry_CommonTopologies_StackGeomDet_H

#include "Geometry/CommonTopologies/interface/DoubleSensGeomDet.h"
#include "DataFormats/DetId/interface/DetId.h"

class StackGeomDet : public DoubleSensGeomDet {
public:
  StackGeomDet(BoundPlane* sp, const GeomDetUnit* lowerDet, const GeomDetUnit* upperDet, const DetId stackDetId)
      : DoubleSensGeomDet(sp, lowerDet, upperDet, stackDetId) {}

  ~StackGeomDet() override = default;

  const GeomDetUnit* lowerDet() const { return firstDet(); };
  const GeomDetUnit* upperDet() const { return secondDet(); };
};

#endif
