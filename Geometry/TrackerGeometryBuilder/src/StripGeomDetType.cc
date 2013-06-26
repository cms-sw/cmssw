#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"


void StripGeomDetType::setTopology( TopologyType* topol) 
{
  if (topol != theTopology) {
    delete theTopology;
    theTopology = topol;
  }
}
