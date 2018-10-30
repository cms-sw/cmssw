#include "Geometry/MTDGeometryBuilder/interface/MTDGeomDetUnit.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeomDetType.h"

#include "Geometry/CommonTopologies/interface/SurfaceDeformation.h"

MTDGeomDetUnit::MTDGeomDetUnit( BoundPlane* sp, MTDGeomDetType const * type, DetId id) : 
  MTDGeomDet(sp), theTopology(new ProxyMTDTopology(type, sp))
{
  setDetId(id);
}

const GeomDetType& MTDGeomDetUnit::type() const { return theTopology->type(); }

const MTDGeomDetType& MTDGeomDetUnit::specificType() const { return theTopology->specificType(); }

const Topology& MTDGeomDetUnit::topology() const { return *theTopology; }

const PixelTopology& MTDGeomDetUnit::specificTopology() const { return *theTopology; }

void MTDGeomDetUnit::setSurfaceDeformation(const SurfaceDeformation * deformation)
{
  theTopology->setSurfaceDeformation(deformation);
}
