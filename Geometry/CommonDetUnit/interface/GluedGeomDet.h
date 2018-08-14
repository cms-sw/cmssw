#ifndef Geometry_TrackerGeometryBuilder_GluedGeomDet_H
#define Geometry_TrackerGeometryBuilder_GluedGeomDet_H

#include "Geometry/CommonDetUnit/interface/TrackerGeomDet.h"
#include "DataFormats/DetId/interface/DetId.h"

class GluedGeomDet final : public TrackerGeomDet {
public:

  GluedGeomDet( BoundPlane* sp, const GeomDetUnit* monoDet,  const GeomDetUnit* stereoDet, DetId gluedDetId);
  
  ~GluedGeomDet() override;

  bool isLeaf() const override { return false;}
  std::vector<const GeomDet*> components() const override;

  // Which subdetector
  SubDetector subDetector() const override {return theMonoDet->subDetector();}

  const GeomDetUnit* monoDet() const { return theMonoDet;}
  const GeomDetUnit* stereoDet() const { return theStereoDet;}

private:
  const GeomDetUnit* theMonoDet;
  const GeomDetUnit* theStereoDet;  
};

#endif
