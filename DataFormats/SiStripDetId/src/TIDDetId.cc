#include "DataFormats/SiStripDetId/interface/TIDDetId.h"


std::ostream& operator<<(std::ostream& os,const TIDDetId& id) {
  unsigned int         theDisk   = id.wheel();
  unsigned int         theRing   = id.ring();
  std::vector<unsigned int> theModule = id.module();
  std::string side;
  std::string part;
  side = (id.side() == 1 ) ? "-" : "+";
  part = (theModule[0] == 1 ) ? "back" : "front";
  std::string type;
  type = (id.stereo() == 0) ? "r-phi" : "stereo";
  type = (id.glued() == 0) ? type : type+" glued";
  type = (id.isDoubleSide()) ? "double side" : type;
  return os << "TID" << side
	    << " Disk " << theDisk
	    << " Ring " << theRing << " " << part
	    << " Module " << theModule[1] << " " << type
	    << " (" << id.rawId() << ")";
}

