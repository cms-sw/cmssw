#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "Geometry/Surface/interface/Bounds.h"


#include "CLHEP/Units/PhysicalConstants.h"


StripGeomDetUnit::StripGeomDetUnit( BoundPlane* sp, StripGeomDetType*& type,const GeometricDet*& gd) : 
   GeomDetUnit(sp),theType( type), theGD(gd)
{}



GeomDetType& StripGeomDetUnit::type() const { return *theType;}


const Topology& StripGeomDetUnit::topology() const {return specificType().topology();}

const StripTopology& StripGeomDetUnit::specificTopology() const { 
  return specificType().specificTopology();
}

DetId StripGeomDetUnit::geographicalId() const {return theGD->geographicalID();}
