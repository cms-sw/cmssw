#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

GeomDetUnit::GeomDetUnit( BoundPlane* sp) : thePlane(sp)
{}

GeomDetUnit::~GeomDetUnit()
{}

const BoundSurface& GeomDetUnit::surface() const 
{
    return *thePlane;
}
