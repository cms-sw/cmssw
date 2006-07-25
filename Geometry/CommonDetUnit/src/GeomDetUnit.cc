#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"

GeomDetUnit::GeomDetUnit( BoundPlane* sp) : GeomDet(sp)
{}

GeomDetUnit::GeomDetUnit( const ReferenceCountingPointer<BoundPlane>& plane) :
  GeomDet(plane) {}

GeomDetUnit::~GeomDetUnit()
{}

GeomDet::SubDetector GeomDetUnit::subDetector() const {
  return type().subDetector();
}

