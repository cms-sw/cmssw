#ifndef Geometry_CommonTopologies_StackGeomDet_H
#define Geometry_CommonTopologies_StackGeomDet_H

#include "Geometry/CommonTopologies/interface/TrackerGeomDet.h"
#include "DataFormats/DetId/interface/DetId.h"

class StackGeomDet : public TrackerGeomDet {
public:
  StackGeomDet(BoundPlane* sp, const GeomDetUnit* lowerDet, const GeomDetUnit* upperDet, const DetId stackDetId);

  ~StackGeomDet() override;

  bool isLeaf() const override { return false; }
  std::vector<const GeomDet*> components() const override;

  // Which subdetector
  SubDetector subDetector() const override { return theLowerDet->subDetector(); };

  const GeomDetUnit* lowerDet() const { return theLowerDet; };
  const GeomDetUnit* upperDet() const { return theUpperDet; };

private:
  const GeomDetUnit* theLowerDet;
  const GeomDetUnit* theUpperDet;
};

#endif
