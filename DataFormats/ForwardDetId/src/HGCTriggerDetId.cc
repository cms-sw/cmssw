#include "DataFormats/ForwardDetId/interface/HGCTriggerDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <ostream>
#include <iostream>

const HGCTriggerDetId HGCTriggerDetId::Undefined(ForwardEmpty,0,0,0,0,0);

HGCTriggerDetId::HGCTriggerDetId() : DetId() {
}

HGCTriggerDetId::HGCTriggerDetId(uint32_t rawid) : DetId(rawid) {
}

HGCTriggerDetId::HGCTriggerDetId(ForwardSubdetector subdet, int zp, int lay, int sec, int subsec, int cell) : DetId(Forward,subdet) {  

  id_ |= (( cell & cell_mask) << cell_shift) ;
  id_ |= (( sec & sector_mask)<< sector_shift);
  if(subsec<0) subsec=0;
  id_ |= (( subsec & subsector_mask) <<subsector_shift);
  id_ |= (( lay & layer_mask) <<layer_shift);
  if( zp<0) zp = 0;
  id_ |= (( zp & zside_mask) <<zside_shift); 
}

HGCTriggerDetId::HGCTriggerDetId(const DetId& gen) {
  if (!gen.null()) {
    ForwardSubdetector subdet=(ForwardSubdetector(gen.subdetId()));
    if (gen.det()!=Forward || (subdet!=HGCTrigger)) {
      throw cms::Exception("Invalid DetId") << "Cannot initialize HGCTriggerDetId from " << std::hex << gen.rawId() << std::dec; 
    }  
  }
  id_ = gen.rawId();
}

HGCTriggerDetId& HGCTriggerDetId::operator=(const DetId& gen) {
  if (!gen.null()) {
    ForwardSubdetector subdet=(ForwardSubdetector(gen.subdetId()));
    if (gen.det()!=Forward || (subdet!=HGCTrigger)) {
      throw cms::Exception("Invalid DetId") << "Cannot assign HGCTriggerDetId from " << std::hex << gen.rawId() << std::dec; 
    }  
  }
  id_ = gen.rawId();
  return (*this);
}

std::ostream& operator<<(std::ostream& s,const HGCTriggerDetId& id) {
  switch (id.subdet()) {
  case(HGCTrigger) : return s << "isEE=" << id.isEE() 
			 << " zpos=" << id.zside() 
			 << " layer=" << id.layer() 
			 << " phi subSector=" << id.subsector()
			 << " sector=" << id.sector() 
			 << " cell=" << id.cell();
  default : return s << id.rawId();
  }
}


