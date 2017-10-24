#ifndef Geometry_TrackerGeometryBuilder_GluedGeomDet_H
#define Geometry_TrackerGeometryBuilder_GluedGeomDet_H

#include "Geometry/CommonDetUnit/interface/TrackerGeomDet.h"
#include "DataFormats/DetId/interface/DetId.h"

class GluedGeomDet final : public TrackerGeomDet {
public:

  GluedGeomDet( BoundPlane* sp, const std::shared_ptr< GeomDet > monoDet,
		const std::shared_ptr< GeomDet > stereoDet, DetId gluedDetId );
  
  ~GluedGeomDet() override;

  bool isLeaf() const override { return false;}
  std::vector< std::shared_ptr< GeomDet >> components() const override;

  // Which subdetector
  SubDetector subDetector() const override {return theMonoDet->subDetector();}

  const std::shared_ptr< GeomDet > monoDet() const { return theMonoDet;}
  const std::shared_ptr< GeomDet > stereoDet() const { return theStereoDet;}

private:
  const std::shared_ptr< GeomDet > theMonoDet;
  const std::shared_ptr< GeomDet > theStereoDet;  
};

#endif
