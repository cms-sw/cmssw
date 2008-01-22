#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"




PixelGeomDetUnit::PixelGeomDetUnit( BoundPlane* sp, PixelGeomDetType* type,const GeometricDet* gd): GeomDetUnit(sp),
												 theType(type),theGD(gd)
{}


const GeomDetType& PixelGeomDetUnit::type() const { return *theType;}


const Topology& PixelGeomDetUnit::topology() const {return specificType().topology();}

const PixelTopology& PixelGeomDetUnit::specificTopology() const { 
  return specificType().specificTopology();
}

DetId PixelGeomDetUnit::geographicalId() const {return theGD->geographicalID();}
