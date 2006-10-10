#include "DataFormats/SiStripDetId/interface/TIDDetId.h"

TIDDetId::TIDDetId() : SiStripDetId() {
}
TIDDetId::TIDDetId(uint32_t rawid) : SiStripDetId(rawid) {
}
TIDDetId::TIDDetId(const DetId& id) : SiStripDetId(id.rawId()) {
}

std::ostream& operator<<(std::ostream& os,const TIDDetId& id) {
  return os << "(TID " 
    //	     << id.whell() << ',' 
    //	     << id.ring()  << ',' 
    //	     << id.det()   << ',' 
    //	     << id.ster()  <<')';
	   <<')';
}
