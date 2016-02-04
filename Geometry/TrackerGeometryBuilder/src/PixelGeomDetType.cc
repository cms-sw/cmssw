#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetType.h"


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
