#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <ostream>

const HGCEEDetId HGCEEDetId::Undefined(ForwardEmpty,0,0,0,0,0);

HGCEEDetId::HGCEEDetId() : DetId() {
}

HGCEEDetId::HGCEEDetId(uint32_t rawid) : DetId(rawid) {
}

HGCEEDetId::HGCEEDetId(ForwardSubdetector subdet, int zp, int lay, int mod,
		       int cellx, int celly) : DetId(Forward,subdet) {
  // (no checking at this point!)
  id_ |= (((zp>0) ? 0x1000000 : 0) | ((lay&0x3F)<<18) | ((mod&0x3F)<12) |
	  ((cellx&0x3F)<<6) | (celly&0x3F));
}

HGCEEDetId::HGCEEDetId(const DetId& gen) {
  if (!gen.null()) {
    ForwardSubdetector subdet=(ForwardSubdetector(gen.subdetId()));
    if (gen.det()!=Forward || (subdet!=HGCEE)) {
      throw cms::Exception("Invalid DetId") << "Cannot initialize HGCEEDetId from " << std::hex << gen.rawId() << std::dec; 
    }  
  }
  id_ = gen.rawId();
}

HGCEEDetId& HGCEEDetId::operator=(const DetId& gen) {
  if (!gen.null()) {
    ForwardSubdetector subdet=(ForwardSubdetector(gen.subdetId()));
    if (gen.det()!=Forward || (subdet!=HGCEE)) {
      throw cms::Exception("Invalid DetId") << "Cannot assign HGCEEDetId from " << std::hex << gen.rawId() << std::dec; 
    }  
  }
  id_ = gen.rawId();
  return (*this);
}

std::ostream& operator<<(std::ostream& s,const HGCEEDetId& id) {
  switch (id.subdet()) {
  case(HGCEE) : return s << "(HGCEE " << id.zside() << ',' << id.layer() << ',' << id.module() << ',' << id.cellX() << ',' << id.cellY() << ')';
  default : return s << id.rawId();
  }
}


