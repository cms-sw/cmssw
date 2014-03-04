#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <ostream>

const HGCHEDetId HGCHEDetId::Undefined(ForwardEmpty,0,0,0,0,0);

HGCHEDetId::HGCHEDetId() : DetId() {
}

HGCHEDetId::HGCHEDetId(uint32_t rawid) : DetId(rawid) {
}

HGCHEDetId::HGCHEDetId(ForwardSubdetector subdet, int zp, int lay, int mod, int subsec, int cell) : DetId(Forward,subdet) 
{  
  uint32_t rawid=0;
  rawid |= ((cell   & 0xffff) << 0 );
  rawid |= ((mod    & 0x1f)   << 16);
  rawid |= ((subsec & 0x1)    << 21);
  rawid |= ((lay    & 0x1f)   << 22);
  if(zp>0) rawid |= ((zp     & 0x1)    << 27);
  //rawid |= (0                 << 28);
  rawid |= ((Forward & 0x7)   << 29);
  id_=rawid;
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


