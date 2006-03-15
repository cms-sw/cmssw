#include <typeinfo>
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include <algorithm>
#include <iostream>
#include <map>

GluedGeomDet::GluedGeomDet( BoundPlane* sp,const GeomDetUnit* monoDet, const GeomDetUnit* stereoDet) : 
  GeomDet(sp),theMonoDet(monoDet),theStereoDet(stereoDet){
  child.push_back(theMonoDet);
  child.push_back(theStereoDet);
}

GluedGeomDet::~GluedGeomDet()
{}

DetId GluedGeomDet::geographicalId() const {
  StripSubdetector subdet(theMonoDet->geographicalId().rawId());
  return DetId(subdet.glued());
}

std::vector<const GeomDet*> GluedGeomDet::components() const {
  return child;
}
