#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <ostream>

const HGCHEDetId HGCHEDetId::Undefined(ForwardEmpty,0,0,0,0,0);

HGCHEDetId::HGCHEDetId() : DetId() {
}

HGCHEDetId::HGCHEDetId(uint32_t rawid) : DetId(rawid) {
}

HGCHEDetId::HGCHEDetId(ForwardSubdetector subdet, int zp, int lay, int mod,
		       int cellx, int celly) : DetId(Forward,subdet) {
  // (no checking at this point!)
  id_ |= (((zp>0) ? 0x1000000 : 0) | ((lay&0x3F)<<18) | ((mod&0x3F)<12) |
	  ((cellx&0x3F)<<6) | (celly&0x3F));
}

HGCHEDetId::HGCHEDetId(const DetId& gen) {
  if (!gen.null()) {
    ForwardSubdetector subdet=(ForwardSubdetector(gen.subdetId()));
    if (gen.det()!=Forward || (subdet!=HGCHE)) {
      throw cms::Exception("Invalid DetId") << "Cannot initialize HGCHEDetId from " << std::hex << gen.rawId() << std::dec; 
    }  
  }
  id_ = gen.rawId();
}

HGCHEDetId& HGCHEDetId::operator=(const DetId& gen) {
  if (!gen.null()) {
    ForwardSubdetector subdet=(ForwardSubdetector(gen.subdetId()));
    if (gen.det()!=Forward || (subdet!=HGCHE)) {
      throw cms::Exception("Invalid DetId") << "Cannot assign HGCHEDetId from " << std::hex << gen.rawId() << std::dec; 
    }  
  }
  id_ = gen.rawId();
  return (*this);
}

std::ostream& operator<<(std::ostream& s,const HGCHEDetId& id) {
  switch (id.subdet()) {
  case(HGCHE) : return s << "(HGCHE " << id.zside() << ',' << id.layer() << ',' << id.module() << ',' << id.cellX() << ',' << id.cellY() << ')';
  default : return s << id.rawId();
  }
}


