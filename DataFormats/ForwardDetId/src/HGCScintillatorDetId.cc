#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <ostream>
#include <iostream>

const HGCScintillatorDetId HGCScintillatorDetId::Undefined(0,0,0,0);

HGCScintillatorDetId::HGCScintillatorDetId() : DetId() {
}

HGCScintillatorDetId::HGCScintillatorDetId(uint32_t rawid) : DetId(rawid) {
}

HGCScintillatorDetId::HGCScintillatorDetId(int type, int layer, int radius,
					   int phi) : DetId(HGCalHSc,ForwardEmpty) {

  int zside      = (radius < 0) ? 1 : 0;
  int radiusAbs  = std::abs(radius);
  id_ |= (((type&kHGCalTypeMask)<<kHGCalTypeOffset) | 
	  ((zside&kHGCalZsideMask)<<kHGCalZsideOffset) |
	  ((layer&kHGCalLayerMask)<<kHGCalLayerOffset) |
	  ((radiusAbs&kHGCalRadiusMask)<<kHGCalRadiusOffset) |
	  ((phi&kHGCalPhiMask)<<kHGCalPhiOffset));
}

HGCScintillatorDetId::HGCScintillatorDetId(const DetId& gen) {
  if (!gen.null()) {
    if (gen.det()!=HGCalHSc) {
      throw cms::Exception("Invalid DetId") << "Cannot initialize HGCScintillatorDetId from " << std::hex << gen.rawId() << std::dec; 
    }  
  }
  id_ = gen.rawId();
}

HGCScintillatorDetId& HGCScintillatorDetId::operator=(const DetId& gen) {
  if (!gen.null()) {
    if (gen.det()!=HGCalHSc) {
      throw cms::Exception("Invalid DetId") << "Cannot assign HGCScintillatorDetId from " << std::hex << gen.rawId() << std::dec; 
    }  
  }
  id_ = gen.rawId();
  return (*this);
}

std::ostream& operator<<(std::ostream& s,const HGCScintillatorDetId& id) {
  return s << " EE:HE= " << id.isEE() << ":" << id.isHE()
	   << " type= " << id.type()  << " layer= " << id.layer() 
	   << " radius= "  << id.iradius() << " phi= " << id.iphi();
}
