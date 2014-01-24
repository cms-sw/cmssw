#include "DataFormats/ForwardDetId/interface/CFCDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <ostream>

const CFCDetId CFCDetId::Undefined(ForwardEmpty,0,0,0,0);

CFCDetId::CFCDetId() : DetId() {
}

CFCDetId::CFCDetId(uint32_t rawid) : DetId(rawid) {
}

CFCDetId::CFCDetId(ForwardSubdetector subdet, int ieta, int iphi, int depth, 
		   int type) : DetId(Forward,subdet) {
  // (no checking at this point!)
  id_ |= ((depth&0x7)<<21) | ((type&0x1)<20) |
    ((ieta>0)?(0x1000000|((ieta&0x3FF)<<10)):(((-ieta)&0x3FF)<<10)) |
    (iphi&0x3FF);
}

CFCDetId::CFCDetId(const DetId& gen) {
  if (!gen.null()) {
    ForwardSubdetector subdet=(ForwardSubdetector(gen.subdetId()));
    if (gen.det()!=Forward || (subdet!=ForwardCFC)) {
      throw cms::Exception("Invalid DetId") << "Cannot initialize CFCDetId from " << std::hex << gen.rawId() << std::dec; 
    }  
  }
  id_ = gen.rawId();
}

CFCDetId& CFCDetId::operator=(const DetId& gen) {
  if (!gen.null()) {
    ForwardSubdetector subdet=(ForwardSubdetector(gen.subdetId()));
    if (gen.det()!=Forward || (subdet!=ForwardCFC)) {
      throw cms::Exception("Invalid DetId") << "Cannot assign CFCDetId from " << std::hex << gen.rawId() << std::dec; 
    }  
  }
  id_ = gen.rawId();
  return (*this);
}

std::ostream& operator<<(std::ostream& s,const CFCDetId& id) {
  switch (id.subdet()) {
  case(ForwardCFC) : return s << "(CFC " << id.ieta() << ',' << id.iphi() << ',' << id.depth() << ',' << id.type() << ')';
  default : return s << id.rawId();
  }
}


