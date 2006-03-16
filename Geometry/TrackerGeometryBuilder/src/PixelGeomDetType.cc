#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

PixelGeomDetType::PixelGeomDetType(TopologyType* t,std::string& name,SubDetector& det) : GeomDetType(name,det),
    theTopology(t){}

const Topology&  PixelGeomDetType::topology()  const
{
    return *theTopology;
}

const PixelTopology& PixelGeomDetType::specificTopology() const
{
    return *theTopology;
}


void PixelGeomDetType::setTopology( TopologyType* topol) 
{
  if (topol != theTopology) {
    delete theTopology;
    theTopology = topol;
  }
}
