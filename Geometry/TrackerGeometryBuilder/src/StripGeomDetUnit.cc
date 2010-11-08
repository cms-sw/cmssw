#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"

#include "Geometry/CommonTopologies/interface/SurfaceDeformation.h"


StripGeomDetUnit::StripGeomDetUnit( BoundPlane* sp, StripGeomDetType* type,const GeometricDet* gd) : 
   GeomDetUnit(sp),theType( type), theGD(gd)
{
  setDetId(theGD->geographicalID());
}



const GeomDetType& StripGeomDetUnit::type() const { return *theType;}


const Topology& StripGeomDetUnit::topology() const {return specificType().topology();}

const StripTopology& StripGeomDetUnit::specificTopology() const { 
  return specificType().specificTopology();
}

void StripGeomDetUnit::setSurfaceDeformation(const SurfaceDeformation * deformation)
{
  theSurfaceDeformation = deformation;
}
