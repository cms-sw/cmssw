#include "DataFormats/ForwardDetId/interface/FastTimeDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <ostream>
#include <iostream>

const FastTimeDetId FastTimeDetId::Undefined(0x60000000u);

FastTimeDetId::FastTimeDetId() : DetId() { }

FastTimeDetId::FastTimeDetId(uint32_t rawid) : DetId(rawid) { }

FastTimeDetId::FastTimeDetId(int module_ix, int module_iy, int iz) : DetId(Forward,FastTime) {
  id_ |= (((module_iy&kFastTimeCellYMask)<<kFastTimeCellYOffset) | 
	  ((module_ix&kFastTimeCellXMask)<<kFastTimeCellXOffset) | 
	  ((iz>0)?(0x10000):(0)));
}

FastTimeDetId::FastTimeDetId(const DetId& gen) {
  if (!gen.null()) {
    ForwardSubdetector subdet=(ForwardSubdetector(gen.subdetId()));
    if (gen.det()!=Forward || (subdet!=FastTime)) {
      throw cms::Exception("Invalid DetId") << "Cannot initialize FastTimeDetId from " << std::hex << gen.rawId() << std::dec; 
    }  
  }
  id_ = gen.rawId();
}

FastTimeDetId& FastTimeDetId::operator=(const DetId& gen) {
  if (!gen.null()) {
    ForwardSubdetector subdet=(ForwardSubdetector(gen.subdetId()));
    if (gen.det()!=Forward || (subdet!=FastTime)) {
      throw cms::Exception("Invalid DetId") << "Cannot assign FastTimeDetId from " << std::hex << gen.rawId() << std::dec; 
    }  
  }
  id_ = gen.rawId();
  return (*this);
}

std::ostream& operator<<(std::ostream& s,const FastTimeDetId& id) {
  return s << "(FastTime iz "<< ((id.zside()>0)?("+ "):("- "))
	   << " ix " << id.ix() << ", iy " << id.iy() << ')';
}
