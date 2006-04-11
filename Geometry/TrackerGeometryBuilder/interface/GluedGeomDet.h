#ifndef Geometry_TrackerGeometryBuilder_GluedGeomDet_H
#define Geometry_TrackerGeometryBuilder_GluedGeomDet_H

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"

class GluedGeomDet : public GeomDet{
public:

  GluedGeomDet( BoundPlane* sp, const GeomDetUnit* monoDet,  const GeomDetUnit* stereoDet);
  
  virtual ~GluedGeomDet();

  virtual DetId geographicalId() const;
  virtual std::vector<const GeomDet*> components() const;

  const GeomDetUnit* monoDet() const { return theMonoDet;}
  const GeomDetUnit* stereoDet() const { return theStereoDet;}

private:
  const GeomDetUnit* theMonoDet;
  const GeomDetUnit* theStereoDet;  
  std::vector<const GeomDet*> child;
};

#endif
