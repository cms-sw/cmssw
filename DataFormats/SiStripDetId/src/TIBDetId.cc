#include "DataFormats/SiStripDetId/interface/TIBDetId.h"

namespace cms
{

  TIBDetId::TIBDetId() : DetId() {
  }
  
  TIBDetId::TIBDetId(uint32_t rawid) : DetId(rawid) {
  }
  

 std::ostream& operator<<(std::ostream& s,const TIBDetId& id) {
    return s << "(TIB " 
      //	     << id.layer() << ',' 
      //	     << id.strng() << ',' 
      //	     << id.det() << ',' 
      //	     << id.ster() <<')';
	     <<')';
  }
  
}
