#include "DataFormats/SiStripDetId/interface/TECDetId.h"

#include <iostream>


std::ostream& operator<<(std::ostream& os,const TECDetId& id) {
  unsigned int              theWheel  = id.wheel();
  unsigned int              theModule = id.module();
  std::vector<unsigned int> thePetal  = id.petal();
  unsigned int              theRing   = id.ring();
  std::string side;
  std::string petal;
  side  = (id.side() == 1 ) ? "-" : "+";
  petal = (thePetal[0] == 1 ) ? "back" : "front";
  std::string type;
  type = (id.stereo() == 0) ? "r-phi" : "stereo";
  type = (id.glued() == 0) ? type : type+" glued";
  type = (id.isDoubleSide()) ? "double side" : type;
  return os << "TEC" << side
	    << " Wheel " << theWheel
      	    << " Petal " << thePetal[1] << " " << petal
	    << " Ring " << theRing
	    << " Module " << theModule << " " << type
	    << " (" << id.rawId() << ")";
}
