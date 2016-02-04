#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"

#include "Geometry/CommonTopologies/interface/SurfaceDeformation.h"

StripGeomDetUnit::StripGeomDetUnit( BoundPlane* sp, StripGeomDetType* type,const GeometricDet* gd) : 
  GeomDetUnit(sp), theTopology(new ProxyStripTopology(type, sp)), theGD(gd)
{
  setDetId(theGD->geographicalID());
}

const GeomDetType& StripGeomDetUnit::type() const { return theTopology->type(); }

StripGeomDetType& StripGeomDetUnit::specificType() const { return theTopology->specificType(); }

const Topology& StripGeomDetUnit::topology() const { return *theTopology; }

const StripTopology& StripGeomDetUnit::specificTopology() const { return *theTopology; }

void StripGeomDetUnit::setSurfaceDeformation(const SurfaceDeformation * deformation)
{
  theTopology->setSurfaceDeformation(deformation);
}
