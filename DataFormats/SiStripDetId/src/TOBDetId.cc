#include "DataFormats/SiStripDetId/interface/TOBDetId.h"

TOBDetId::TOBDetId() : DetId() {
}

TOBDetId::TOBDetId(uint32_t rawid) : DetId(rawid) {
}

std::ostream& operator<<(std::ostream& os,const TOBDetId& id) {
  return os << "(TOB " 
    //	     << id.layer() << ',' 
    //	     << id.rod()   << ',' 
    //	     << id.det()   << ',' 
    //	     << id.ster()  <<')';
	   <<')';
}

