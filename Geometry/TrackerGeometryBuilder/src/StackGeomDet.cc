#include "Geometry/TrackerGeometryBuilder/interface/StackGeomDet.h"

StackGeomDet::StackGeomDet( BoundPlane* sp, std::shared_ptr< GeomDet > lowerDet, std::shared_ptr< GeomDet > upperDet, const DetId stackDetId) : 
  TrackerGeomDet(sp),theLowerDet(lowerDet),theUpperDet(upperDet){
  setDetId(stackDetId);
}

StackGeomDet::~StackGeomDet()
{}

std::vector< std::shared_ptr< GeomDet >> StackGeomDet::components() const {
  return std::vector< std::shared_ptr< GeomDet >>{theLowerDet,theUpperDet};
}
