#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

GeomDetUnit::GeomDetUnit( BoundPlane* sp) : GeomDet(sp)
{}

GeomDetUnit::GeomDetUnit( const ReferenceCountingPointer<BoundPlane>& plane) :
  GeomDet(plane) {}

GeomDetUnit::~GeomDetUnit()
{}
