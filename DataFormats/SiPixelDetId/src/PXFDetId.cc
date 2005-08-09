#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

namespace cms
{

  PXFDetId::PXFDetId() : DetId() {
  }
  
  PXFDetId::PXFDetId(uint32_t rawid) : DetId(rawid) {
  }
  
  std::ostream& operator<<(std::ostream& s,const PXFDetId& id) {
    return s << "(PixelEndcap " 
	     << id.disk() << ',' 
	     << id.blade()  << ',' 
	     << id.det()   << ')'; 
  }
  
}
