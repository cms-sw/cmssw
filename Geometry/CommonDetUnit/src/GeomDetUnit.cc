#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "FWCore/Utilities/interface/Exception.h"

GeomDetUnit::GeomDetUnit( Plane* sp) : GeomDet(sp)
{}

GeomDetUnit::GeomDetUnit( const ReferenceCountingPointer<Plane>& plane) :
  GeomDet(plane) {}

GeomDetUnit::~GeomDetUnit()
{}

GeomDet::SubDetector GeomDetUnit::subDetector() const {
  return type().subDetector();
}

void GeomDetUnit::setSurfaceDeformation(const SurfaceDeformation * /*deformation*/)
{
  throw cms::Exception("Geometry") 
    << "setting SurfaceDeformation not implemented for DetId "
    << geographicalId().rawId() << " det="
    << geographicalId().det() << " subdetId="
    << geographicalId().subdetId();
}
