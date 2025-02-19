#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"

PXBDetId::PXBDetId() : DetId() {
}

PXBDetId::PXBDetId(uint32_t rawid) : DetId(rawid) {
}
PXBDetId::PXBDetId(const DetId& id) : DetId(id.rawId()) {
}


std::ostream& operator<<(std::ostream& os,const PXBDetId& id) {
  return os << "(PixelBarrel " 
	   << id.layer() << ',' 
	   << id.ladder() << ',' 
	   << id.module() << ')'; 
}
