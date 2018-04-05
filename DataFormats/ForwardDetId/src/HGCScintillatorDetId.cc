#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <ostream>
#include <iostream>

const HGCScintillatorDetId HGCScintillatorDetId::Undefined(0,0,0,0);

HGCScintillatorDetId::HGCScintillatorDetId() : DetId() {
}

HGCScintillatorDetId::HGCScintillatorDetId(uint32_t rawid) : DetId(rawid) {
}

HGCScintillatorDetId::HGCScintillatorDetId(int type, int layer, int eta,
					   int phi) : DetId(HGCalHSc,HGCHEB) {

  id_ |= ((type&kHGCalTypeMask)<<kHGCalTypeOffset) | 
    ((layer&kHGCalLayerMask)<<kHGCalLayerOffset) |
    ((eta<0)?(kHGCalZsideMask2|(-eta<<kHGCalEtaOffset)):(eta<<kHGCalEtaOffset)) |
    ((phi&kHGCalPhiMask)<<kHGCalPhiOffset);
}

HGCScintillatorDetId::HGCScintillatorDetId(const DetId& gen) {
  if (!gen.null()) {
    if ((gen.det()!=HGCalHSc) || 
	(ForwardSubdetector)(gen.subdetId()!=HGCHEB)) {
      throw cms::Exception("Invalid DetId") << "Cannot initialize HGCScintillatorDetId from " << std::hex << gen.rawId() << std::dec; 
    }  
  }
  id_ = gen.rawId();
}

HGCScintillatorDetId& HGCScintillatorDetId::operator=(const DetId& gen) {
  if (!gen.null()) {
    if ((gen.det()!=HGCalHSc) || 
	(ForwardSubdetector)(gen.subdetId()!=HGCHEB)) {
      throw cms::Exception("Invalid DetId") << "Cannot assign HGCScintillatorDetId from " << std::hex << gen.rawId() << std::dec; 
    }  
  }
  id_ = gen.rawId();
  return (*this);
}

std::ostream& operator<<(std::ostream& s,const HGCScintillatorDetId& id) {
  return s << " EE:HE= " << id.isEE() << ":" << id.isHE()
	   << " type= " << id.type()  << " layer= " << id.layer() 
	   << " eta= "  << id.ieta() << " phi= " << id.iphi();
}


