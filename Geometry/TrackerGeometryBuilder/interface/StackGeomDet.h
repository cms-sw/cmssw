#ifndef Geometry_TrackerGeometryBuilder_StackGeomDet_H
#define Geometry_TrackerGeometryBuilder_StackGeomDet_H

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"

class StackGeomDet : public GeomDet{
public:

  StackGeomDet(Plane* sp) : GeomDet(sp) {};
  StackGeomDet( Plane* sp, const GeomDetUnit* lowerDet,  const GeomDetUnit* upperDet, const DetId stackDetId);
  
  virtual ~StackGeomDet();

  virtual SubDetector subDetector() const { return theLowerDet->subDetector(); };
  virtual std::vector<const GeomDet*> components() const;

  const GeomDetUnit* lowerDet() const { return theLowerDet; };
  const GeomDetUnit* upperDet() const { return theUpperDet; };

private:
  const GeomDetUnit* theLowerDet;
  const GeomDetUnit* theUpperDet;  
  std::vector<const GeomDet*> child;
};

#endif
