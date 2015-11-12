#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"

GluedGeomDet::GluedGeomDet( BoundPlane* sp,const GeomDetUnit* monoDet, const GeomDetUnit* stereoDet, const DetId gluedDetId) : 
  GeomDet(sp),theMonoDet(monoDet),theStereoDet(stereoDet){
  child.push_back(theMonoDet);
  child.push_back(theStereoDet);
  setDetId(gluedDetId);
}

GluedGeomDet::~GluedGeomDet()
{}

std::vector<const GeomDet*> GluedGeomDet::components() const {
  return child;
}
