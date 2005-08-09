#include "DataFormats/SiStripDetId/interface/TIDDetId.h"

namespace cms
{

  TIDDetId::TIDDetId() : DetId() {
  }
  
  TIDDetId::TIDDetId(uint32_t rawid) : DetId(rawid) {
  }
  
  std::ostream& operator<<(std::ostream& s,const TIDDetId& id) {
    return s << "(TID " 
      //	     << id.whell() << ',' 
      //	     << id.ring()  << ',' 
      //	     << id.det()   << ',' 
      //	     << id.ster()  <<')';
	     <<')';
  }
  
}
