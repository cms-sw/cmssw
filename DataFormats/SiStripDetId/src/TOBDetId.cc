#include "DataFormats/SiStripDetId/interface/TOBDetId.h"

namespace cms
{

  TOBDetId::TOBDetId() : DetId() {
  }
  
  TOBDetId::TOBDetId(uint32_t rawid) : DetId(rawid) {
  }
  
  std::ostream& operator<<(std::ostream& s,const TOBDetId& id) {
    return s << "(TOB " 
      //	     << id.layer() << ',' 
      //	     << id.rod()   << ',' 
      //	     << id.det()   << ',' 
      //	     << id.ster()  <<')';
	     <<')';
  }
  
}
