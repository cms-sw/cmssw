#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"

#include "Geometry/CommonTopologies/interface/SurfaceDeformation.h"


PixelGeomDetUnit::PixelGeomDetUnit( BoundPlane* sp, PixelGeomDetType* type,const GeometricDet* gd): GeomDetUnit(sp),
												 theType(type),theGD(gd)
{
  setDetId(theGD->geographicalID());
}


const GeomDetType& PixelGeomDetUnit::type() const { return *theType;}


const Topology& PixelGeomDetUnit::topology() const {return specificType().topology();}

const PixelTopology& PixelGeomDetUnit::specificTopology() const { 
  return specificType().specificTopology();
}

void PixelGeomDetUnit::setSurfaceDeformation(const SurfaceDeformation * deformation)
{
  theSurfaceDeformation = deformation;
}

