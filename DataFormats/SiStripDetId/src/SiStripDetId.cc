#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include <iomanip>



std::ostream& operator<< ( std::ostream& os, const SiStripDetId& id ) {
  return os << "[SiStripDetId::print]" << std::endl
	    << " rawId       : 0x" 
	    << std::hex << std::setfill('0') << std::setw(8)
	    << id.rawId()
	    << std::dec << std::endl
	    << " bits[0:24]  : " 
	    << std::hex << std::setfill('0') << std::setw(8) 
	    << (0x01FFFFFF & id.rawId())
	    << std::dec << std::endl
	    << " Detector    : " << id.det() << std::endl 
	    << " SubDetector : " << id.subdetId() << std::endl
	    << " reserved    : " << id.reserved();
}

