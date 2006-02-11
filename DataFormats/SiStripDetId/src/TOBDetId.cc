#include "DataFormats/SiStripDetId/interface/TOBDetId.h"

TOBDetId::TOBDetId() : DetId() {
}
TOBDetId::TOBDetId(uint32_t rawid) : DetId(rawid) {
}
TOBDetId::TOBDetId(const DetId& id) : DetId(id.rawId()) {
}

std::ostream& operator<<(std::ostream& os,const TOBDetId& id) {
  return os << "(TOB " 
    //	     << id.layer() << ',' 
    //	     << id.rod()   << ',' 
    //	     << id.det()   << ',' 
    //	     << id.ster()  <<')';
	   <<')';
}

