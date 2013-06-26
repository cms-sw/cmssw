#include "DataFormats/SiStripDetId/interface/TOBDetId.h"


std::ostream& operator<<(std::ostream& os,const TOBDetId& id) {
  unsigned int              theLayer  = id.layer();
  std::vector<unsigned int> theRod    = id.rod();
  unsigned int              theModule = id.module();
  std::string side;
  std::string part;
  side = (theRod[0] == 1 ) ? "-" : "+";
  std::string type;
  type = (id.stereo() == 0) ? "r-phi" : "stereo";
  type = (id.glued() == 0) ? type : type+" glued";
  type = (id.isDoubleSide()) ? "double side" : type;
  return os << "TOB" << side
	    << " Layer " << theLayer
	    << " Rod " << theRod[1]
	    << " Module " << theModule << " " << type
	    << " (" << id.rawId() << ")";
}

