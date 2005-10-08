#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"

PXBDetId::PXBDetId() : DetId() {
}

PXBDetId::PXBDetId(uint32_t rawid) : DetId(rawid) {
}

std::ostream& operator<<(std::ostream& os,const PXBDetId& id) {
  return os << "(PixelBarrel " 
	   << id.layer() << ',' 
	   << id.ladder() << ',' 
	   << id.det() << ')'; 
}
