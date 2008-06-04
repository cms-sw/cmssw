#include "DataFormats/SiStripDetId/interface/TOBDetId.h"

TOBDetId::TOBDetId() : SiStripDetId() {
}
TOBDetId::TOBDetId(uint32_t rawid) : SiStripDetId(rawid) {
}
TOBDetId::TOBDetId(const DetId& id) : SiStripDetId(id.rawId()) {
}

bool TOBDetId::isDoubleSide() const {
  // Double Side: only layers 1 and 2
  if( this->glued() == 0 && ( this->layer() == 1 || this->layer() == 2 ) ) {
    return true;
  } else {
    return false;
  }
}

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

