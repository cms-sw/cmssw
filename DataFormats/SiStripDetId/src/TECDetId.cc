#include "DataFormats/SiStripDetId/interface/TECDetId.h"

TECDetId::TECDetId() : SiStripDetId() {
}

TECDetId::TECDetId(uint32_t rawid) : SiStripDetId(rawid) {
}
TECDetId::TECDetId(const DetId& id) : SiStripDetId(id.rawId()){
}

std::ostream& operator<<(std::ostream& os,const TECDetId& id) {
  return os << "(TEC " 
    //	     << id.whell() << ',' 
    //	     << id.petal()[0] << ',' 
    //	     << id.petal()[1] << ',' 
    //	     << id.ring()  << ',' 
    //	     << id.det()   <<','
    //	     << id.stereo()  <<')';
	   <<')';
}

