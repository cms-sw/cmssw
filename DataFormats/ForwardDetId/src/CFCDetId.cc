#include "DataFormats/ForwardDetId/interface/CFCDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>

const CFCDetId CFCDetId::Undefined(ForwardEmpty,0,0,0,0,0);

CFCDetId::CFCDetId() : DetId() {
}

CFCDetId::CFCDetId(uint32_t rawid) : DetId(rawid) {
}

CFCDetId::CFCDetId(ForwardSubdetector subdet, int module, int ieta, int iphi, 
		   int depth, int type) : DetId(Forward,subdet) {

  // (no checking at this point!)
  id_ |= ((depth&0x7)<<21) | ((type&0x1)<<20) |
    ((ieta>0)?(0x80000|((ieta&0xFF)<<11)):(((-ieta)&0xFF)<<11)) |
    ((module&0x1F)<<6) | ((iphi>0)?(0x20|(iphi&0x1F)):((-iphi&0x1F)));
}

CFCDetId::CFCDetId(const DetId& gen) {
  if (!gen.null()) {
    ForwardSubdetector subdet=(ForwardSubdetector(gen.subdetId()));
    if (gen.det()!=Forward || (subdet!=CFC)) {
      throw cms::Exception("Invalid DetId") << "Cannot initialize CFCDetId from " << std::hex << gen.rawId() << std::dec; 
    }  
  }
  id_ = gen.rawId();
}

CFCDetId& CFCDetId::operator=(const DetId& gen) {
  if (!gen.null()) {
    ForwardSubdetector subdet=(ForwardSubdetector(gen.subdetId()));
    if (gen.det()!=Forward || (subdet!=CFC)) {
      throw cms::Exception("Invalid DetId") << "Cannot assign CFCDetId from " << std::hex << gen.rawId() << std::dec; 
    }  
  }
  id_ = gen.rawId();
  return (*this);
}

std::ostream& operator<<(std::ostream& s,const CFCDetId& id) {
  switch (id.subdet()) {
  case(CFC) : return s << "(CFC " << id.ieta() << ',' << id.module() << ',' << id.iphi() << ',' << id.depth() << ',' << id.type() << ')';
  default : return s << id.rawId();
  }
}


