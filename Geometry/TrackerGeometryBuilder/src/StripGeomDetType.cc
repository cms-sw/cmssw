#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

const Topology&      StripGeomDetType::topology()         const
{
    return *theTopology;
}

const StripTopology& StripGeomDetType::specificTopology() const
{
    return *theTopology;
}

void StripGeomDetType::setTopology( TopologyType* topol) 
{
  if (topol != theTopology) {
    delete theTopology;
    theTopology = topol;
  }
}
