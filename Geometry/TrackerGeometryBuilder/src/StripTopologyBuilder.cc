
#include "Geometry/TrackerGeometryBuilder/interface/StripTopologyBuilder.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"
#include "Geometry/Surface/interface/Bounds.h"



StripTopologyBuilder::StripTopologyBuilder(){}

StripTopology* StripTopologyBuilder::build(const Bounds* bs,double apvnumb,std::string part)
{
  theAPVNumb = apvnumb;

  StripTopology* result;
  if (part == "barrel") {
    result = constructBarrel( bs->length(), bs->width());
  }
  else {
    result = constructForward( bs->length(), bs->width(),bs->widthAtHalfLength());
  }
  return result;
}

StripTopology* StripTopologyBuilder::constructBarrel( float length, float width)
{
  int nstrip = int(128*theAPVNumb);
  float pitch = width/nstrip;
  
  return new RectangularStripTopology(nstrip,pitch,length);
}
 
StripTopology* StripTopologyBuilder::constructForward( float length, float width, float widthAtHalf)
{
  int nstrip = int(128*theAPVNumb);
  float pitch = widthAtHalf/nstrip;
  float rCross = widthAtHalf*length/(2*(width-widthAtHalf));
  return new TrapezoidalStripTopology(nstrip,pitch,length,rCross);
}

