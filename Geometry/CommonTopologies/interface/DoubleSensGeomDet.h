#ifndef Geometry_CommonTopologies_DoubleSensGeomDet_H
#define Geometry_CommonTopologies_DoubleSensGeomDet_H

#include "Geometry/CommonTopologies/interface/TrackerGeomDet.h"
#include "DataFormats/DetId/interface/DetId.h"

class DoubleSensGeomDet : public TrackerGeomDet {
public:
  DoubleSensGeomDet(BoundPlane* sp,
                    const GeomDetUnit* firstDet,
                    const GeomDetUnit* secondDet,
                    const DetId doubleSensDetId);

  ~DoubleSensGeomDet() override;

  bool isLeaf() const override { return false; }
  std::vector<const GeomDet*> components() const override;

  // Which subdetector
  SubDetector subDetector() const override { return theFirstDet->subDetector(); };

  const GeomDetUnit* firstDet() const { return theFirstDet; };
  const GeomDetUnit* secondDet() const { return theSecondDet; };

private:
  const GeomDetUnit* theFirstDet;
  const GeomDetUnit* theSecondDet;
};

#endif
