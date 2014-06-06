#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <ostream>
#include <iostream>

const HGCEEDetId HGCEEDetId::Undefined(ForwardEmpty,0,0,0,0,0);

HGCEEDetId::HGCEEDetId() : DetId() {
}

HGCEEDetId::HGCEEDetId(uint32_t rawid) : DetId(rawid) {
}

HGCEEDetId::HGCEEDetId(ForwardSubdetector subdet, int zp, int lay, int sec, int subsec, int cell) : DetId(Forward,subdet) {  

  id_ |= ((cell   & 0xfff) << 0 );
  id_ |= ((sec    & 0x3f)  << 12);
  id_ |= ((subsec & 0x1)   << 18);
  id_ |= ((lay    & 0x1f)  << 19);
  if (zp>0) id_ |= ((zp & 0x1) << 24);
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
  case(HGCEE) : return s << "isEE=" << id.isEE() 
			 << " zpos=" << id.zside() 
			 << " layer=" << id.layer() 
			 << " phi subSector=" << id.subsector()
			 << " sector=" << id.sector() 
			 << " cell=" << id.cell();
  default : return s << id.rawId();
  }
}


