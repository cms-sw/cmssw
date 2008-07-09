#include "DataFormats/SiStripDetId/interface/TIBDetId.h"

TIBDetId::TIBDetId() : SiStripDetId(){
}
TIBDetId::TIBDetId(uint32_t rawid) : SiStripDetId(rawid){
}
TIBDetId::TIBDetId(const DetId& id) : SiStripDetId(id.rawId()){
}
  
bool TIBDetId::isDoubleSide() const {
  // Double Side: only layers 1 and 2
  if( this->glued() == 0 && ( this->layer() == 1 || this->layer() == 2 ) ) {
    return true;
  } else {
    return false;
  }
}

std::ostream& operator<<(std::ostream& os,const TIBDetId& id) {
  unsigned int              theLayer  = id.layer();
  std::vector<unsigned int> theString = id.string();
  unsigned int              theModule = id.module();
  std::string side;
  std::string part;
  side = (theString[0] == 1 ) ? "-" : "+";
  part = (theString[1] == 1 ) ? "int" : "ext";
  std::string type;
  type = (id.stereo() == 0) ? "r-phi" : "stereo";
  type = (id.glued() == 0) ? type : type+" glued";
  type = (id.isDoubleSide()) ? "double side" : type;
  return os << "TIB" << side
	    << " Layer " << theLayer << " " << part
	    << " String " << theString[2]
	    << " Module " << theModule << " " << type
	    << " (" << id.rawId() << ")";
}

