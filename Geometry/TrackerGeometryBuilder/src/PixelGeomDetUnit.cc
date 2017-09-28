#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"

#include "Geometry/CommonTopologies/interface/SurfaceDeformation.h"

PixelGeomDetUnit::PixelGeomDetUnit( BoundPlane* sp, std::shared_ptr< PixelGeomDetType > type, DetId id) : 
  TrackerGeomDet(sp), theTopology(new ProxyPixelTopology(type, sp))
{
  setDetId(id);
}

const GeomDetType& PixelGeomDetUnit::type() const { return theTopology->type(); }

const PixelGeomDetType& PixelGeomDetUnit::specificType() const { return theTopology->specificType(); }

const Topology& PixelGeomDetUnit::topology() const { return *theTopology; }

const PixelTopology& PixelGeomDetUnit::specificTopology() const { return *theTopology; }

void PixelGeomDetUnit::setSurfaceDeformation(const SurfaceDeformation * deformation)
{
  theTopology->setSurfaceDeformation(deformation);
}
