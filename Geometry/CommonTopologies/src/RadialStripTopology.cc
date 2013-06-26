#include <Geometry/CommonTopologies/interface/RadialStripTopology.h>
#include "FWCore/Utilities/interface/Exception.h"

float RadialStripTopology::pitch() const {  
  throw cms::Exception("RadialStripTopology") << "pitch() called - makes no sense, use localPitch(.) instead."; 
  return 0.;
}


std::ostream & operator<<( std::ostream & os, const RadialStripTopology & rst ) {
  os  << "RadialStripTopology " << std::endl
      << " " << std::endl
      << "number of strips          " << rst.nstrips() << std::endl
      << "centre to whereStripsMeet " << rst.centreToIntersection() << std::endl
      << "detector height in y      " << rst.detHeight() << std::endl
      << "angular width of strips   " << rst.phiPitch() << std::endl
      << "phi of one edge           " << rst.phiOfOneEdge() << std::endl
      << "y axis orientation        " << rst.yAxisOrientation() << std::endl
      << "y of centre of strip plane " << rst.yCentreOfStripPlane() << std::endl;
  return os;
}
