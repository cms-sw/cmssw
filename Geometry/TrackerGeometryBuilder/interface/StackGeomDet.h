#ifndef Geometry_TrackerGeometryBuilder_StackGeomDet_H
#define Geometry_TrackerGeometryBuilder_StackGeomDet_H

#include "Geometry/CommonDetUnit/interface/TrackerGeomDet.h"
#include "DataFormats/DetId/interface/DetId.h"

class StackGeomDet : public TrackerGeomDet{
public:

  StackGeomDet( BoundPlane* sp, const GeomDetUnit* lowerDet,  const GeomDetUnit* upperDet, const DetId stackDetId);
  
  virtual ~StackGeomDet();

  bool isLeaf() const override { return false;}
  virtual std::vector<const GeomDet*> components() const;

  // Which subdetector
  virtual SubDetector subDetector() const { return theLowerDet->subDetector(); };

  const GeomDetUnit* lowerDet() const { return theLowerDet; };
  const GeomDetUnit* upperDet() const { return theUpperDet; };

private:
  const GeomDetUnit* theLowerDet;
  const GeomDetUnit* theUpperDet;  
};

#endif
