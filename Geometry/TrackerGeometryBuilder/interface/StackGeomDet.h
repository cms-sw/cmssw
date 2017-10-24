#ifndef Geometry_TrackerGeometryBuilder_StackGeomDet_H
#define Geometry_TrackerGeometryBuilder_StackGeomDet_H

#include "Geometry/CommonDetUnit/interface/TrackerGeomDet.h"
#include "DataFormats/DetId/interface/DetId.h"

class StackGeomDet : public TrackerGeomDet{
public:

  StackGeomDet( BoundPlane* sp, std::shared_ptr< GeomDet > lowerDet,  std::shared_ptr< GeomDet > upperDet, const DetId stackDetId);
  
  ~StackGeomDet() override;

  bool isLeaf() const override { return false;}
  std::vector< std::shared_ptr< GeomDet >> components() const override;

  // Which subdetector
  SubDetector subDetector() const override { return theLowerDet->subDetector(); };

  std::shared_ptr< GeomDet > lowerDet() const { return theLowerDet; };
  std::shared_ptr< GeomDet > upperDet() const { return theUpperDet; };

private:
  std::shared_ptr< GeomDet > theLowerDet;
  std::shared_ptr< GeomDet > theUpperDet;  
};

#endif
