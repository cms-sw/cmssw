#include "Geometry/TrackerGeometryBuilder/interface/StackGeomDet.h"

StackGeomDet::StackGeomDet( Plane* sp, const GeomDetUnit* lowerDet, const GeomDetUnit* upperDet, const DetId stackDetId) : 
  GeomDet(sp),theLowerDet(lowerDet),theUpperDet(upperDet){
  child.push_back(theLowerDet);
  child.push_back(theUpperDet);
  setDetId(stackDetId);
}

StackGeomDet::~StackGeomDet()
{}

std::vector<const GeomDet*> StackGeomDet::components() const {
  return child;
}

