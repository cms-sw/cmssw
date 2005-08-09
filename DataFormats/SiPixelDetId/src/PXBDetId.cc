#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"

namespace cms
{

  PXBDetId::PXBDetId() : DetId() {
  }
  
  PXBDetId::PXBDetId(uint32_t rawid) : DetId(rawid) {
  }
  
  std::ostream& operator<<(std::ostream& s,const PXBDetId& id) {
    return s << "(PixelBarrel " 
	     << id.layer() << ',' 
	     << id.ladder() << ',' 
	     << id.det() << ')'; 
  }
  
}
