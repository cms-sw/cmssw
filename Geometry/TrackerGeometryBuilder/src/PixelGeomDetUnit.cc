#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"

#include "Geometry/CommonTopologies/interface/SurfaceDeformation.h"

PixelGeomDetUnit::PixelGeomDetUnit( BoundPlane* sp, PixelGeomDetType* type,const GeometricDet* gd) : 
  GeomDetUnit(sp), theTopology(new ProxyPixelTopology(type, sp)), theGD(gd)
{
  setDetId(theGD->geographicalID());
}

const GeomDetType& PixelGeomDetUnit::type() const { return theTopology->type(); }

const PixelGeomDetType& PixelGeomDetUnit::specificType() const { return theTopology->specificType(); }

const Topology& PixelGeomDetUnit::topology() const { return *theTopology; }

const PixelTopology& PixelGeomDetUnit::specificTopology() const { return *theTopology; }

void PixelGeomDetUnit::setSurfaceDeformation(const SurfaceDeformation * deformation)
{
  theTopology->setSurfaceDeformation(deformation);
}
