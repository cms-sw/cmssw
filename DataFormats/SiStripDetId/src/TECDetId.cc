#include "DataFormats/SiStripDetId/interface/TECDetId.h"

#include <iostream>

TECDetId::TECDetId() : SiStripDetId() {
}

TECDetId::TECDetId(uint32_t rawid) : SiStripDetId(rawid) {
}
TECDetId::TECDetId(const DetId& id) : SiStripDetId(id.rawId()){
}


bool TECDetId::isDoubleSide() const {
  // Double Side: only rings 1, 2 and 5
  if( this->glued() == 0 && ( this->ring() == 1 || this->ring() == 2 || this->ring() == 5 ) ) {
    return true;
  } else {
    return false;
  }
}

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
