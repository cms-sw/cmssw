#include "DataFormats/ForwardDetId/interface/HGCTriggerHexDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <ostream>
#include <iostream>


const HGCTriggerHexDetId HGCTriggerHexDetId::Undefined(ForwardEmpty,0,0,0,0,0);

HGCTriggerHexDetId::HGCTriggerHexDetId() : DetId() {
}

HGCTriggerHexDetId::HGCTriggerHexDetId(uint32_t rawid) : DetId(rawid) {
}

HGCTriggerHexDetId::
HGCTriggerHexDetId(ForwardSubdetector subdet, int zp, int lay, int wafertype, int mod, int cell) : DetId(Forward,subdet) {  

  if (cell>cell_mask || cell<0 ||
          mod>wafer_mask || mod<0 ||
          wafertype>wafer_type_mask ||
          lay>layer_mask || lay<0)
  {
    zp = lay = wafertype = mod = cell = 0;
  }

  setMaskedId( cell, cell_shift  , cell_mask  );
  setMaskedId( mod , wafer_shift, wafer_mask);
  setMaskedId( wafertype , wafer_type_shift, wafer_type_mask);
  setMaskedId( lay , layer_shift , layer_mask );
  if(zp>0) setMaskedId( zp  , zside_shift , zside_mask ) ;
}

HGCTriggerHexDetId::
HGCTriggerHexDetId(const DetId& gen) {
  if (!gen.null()) {
    if (gen.det()!=Forward) {
      throw cms::Exception("Invalid DetId") << "Cannot initialize HGCTriggerHexDetId from " << std::hex << gen.rawId() << std::dec; 
    }  
  }
  id_ = gen.rawId();
}

HGCTriggerHexDetId& 
HGCTriggerHexDetId::
operator=(const DetId& gen) {
  if (!gen.null()) {
    if (gen.det()!=Forward) {
      throw cms::Exception("Invalid DetId") << "Cannot assign HGCTriggerHexDetId from " << std::hex << gen.rawId() << std::dec; 
    }  
  }
  id_ = gen.rawId();
  return (*this);
}

std::ostream& operator<<(std::ostream& s,const HGCTriggerHexDetId& id) {
  return s << "isHGCal=" << id.isHGCal() << " zpos=" << id.zside() 
	   << " layer=" << id.layer()  << " wafer type=" << id.waferType()
	   << " wafer=" << id.wafer() << " cell=" << id.cell();
}


