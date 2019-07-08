#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"

#include "Geometry/CommonTopologies/interface/SurfaceDeformation.h"

StripGeomDetUnit::StripGeomDetUnit(BoundPlane* sp, StripGeomDetType const* type, DetId id)
    : TrackerGeomDet(sp), theTopology(new ProxyStripTopology(type, sp)) {
  setDetId(id);
}

const GeomDetType& StripGeomDetUnit::type() const { return theTopology->type(); }

const StripGeomDetType& StripGeomDetUnit::specificType() const { return theTopology->specificType(); }

const Topology& StripGeomDetUnit::topology() const { return *theTopology; }

const StripTopology& StripGeomDetUnit::specificTopology() const { return *theTopology; }

void StripGeomDetUnit::setSurfaceDeformation(const SurfaceDeformation* deformation) {
  theTopology->setSurfaceDeformation(deformation);
}
