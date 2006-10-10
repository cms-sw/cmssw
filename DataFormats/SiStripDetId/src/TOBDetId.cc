#include "DataFormats/SiStripDetId/interface/TOBDetId.h"

TOBDetId::TOBDetId() : SiStripDetId() {
}
TOBDetId::TOBDetId(uint32_t rawid) : SiStripDetId(rawid) {
}
TOBDetId::TOBDetId(const DetId& id) : SiStripDetId(id.rawId()) {
}

std::ostream& operator<<(std::ostream& os,const TOBDetId& id) {
  return os << "(TOB " 
    //	     << id.layer() << ',' 
    //	     << id.rod()   << ',' 
    //	     << id.det()   << ',' 
    //	     << id.ster()  <<')';
	   <<')';
}

