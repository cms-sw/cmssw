#include "DataFormats/SiStripDetId/interface/TIDDetId.h"

TIDDetId::TIDDetId() : DetId() {
}
TIDDetId::TIDDetId(uint32_t rawid) : DetId(rawid) {
}
TIDDetId::TIDDetId(const DetId& id) : DetId(id.rawId()) {
}

std::ostream& operator<<(std::ostream& os,const TIDDetId& id) {
  return os << "(TID " 
    //	     << id.whell() << ',' 
    //	     << id.ring()  << ',' 
    //	     << id.det()   << ',' 
    //	     << id.ster()  <<')';
	   <<')';
}
