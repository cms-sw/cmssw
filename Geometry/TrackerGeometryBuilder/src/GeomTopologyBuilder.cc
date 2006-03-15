#include "Geometry/TrackerGeometryBuilder/interface/GeomTopologyBuilder.h"
#include "Geometry/CommonTopologies/interface/Topology.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelTopologyBuilder.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripTopologyBuilder.h"



GeomTopologyBuilder::GeomTopologyBuilder(){}

PixelTopology* GeomTopologyBuilder::buildPixel(const Bounds* bs,double rocRow,double rocCol,double rocInX,double rocInY,std::string part)
{
  PixelTopology* result;
  result = PixelTopologyBuilder().build(bs,rocRow,rocCol,rocInX,rocInY,part);
  return result;
}
StripTopology* GeomTopologyBuilder::buildStrip(const Bounds* bs,double apvnumb,std::string part)
{
  StripTopology* result;
  result = StripTopologyBuilder().build(bs,apvnumb,part);
  return result;
}
