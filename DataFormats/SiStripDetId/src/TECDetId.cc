#include "DataFormats/SiStripDetId/interface/TECDetId.h"

namespace cms
{

  TECDetId::TECDetId() : DetId() {
  }
  
  TECDetId::TECDetId(uint32_t rawid) : DetId(rawid) {
  }
  
  std::ostream& operator<<(std::ostream& s,const TECDetId& id) {
    return s << "(TEC " 
      //	     << id.whell() << ',' 
      //	     << id.petal()[0] << ',' 
      //	     << id.petal()[1] << ',' 
      //	     << id.ring()  << ',' 
      //	     << id.det()   <<','
      //	     << id.stereo()  <<')';
	     <<')';
  }
  
}
