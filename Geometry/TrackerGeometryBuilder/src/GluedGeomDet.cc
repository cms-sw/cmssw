#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

GluedGeomDet::GluedGeomDet( BoundPlane* sp,const GeomDetUnit* monoDet, const GeomDetUnit* stereoDet) : 
  GeomDet(sp),theMonoDet(monoDet),theStereoDet(stereoDet){
  child.push_back(theMonoDet);
  child.push_back(theStereoDet);
  StripSubdetector subdet(theMonoDet->geographicalId().rawId());
  setDetId(subdet.glued());
}

GluedGeomDet::~GluedGeomDet()
{}

std::vector<const GeomDet*> GluedGeomDet::components() const {
  return child;
}
