#include "DataFormats/ForwardDetId/interface/HGCTriggerHexDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <ostream>
#include <iostream>



HGCTriggerHexDetId::HGCTriggerHexDetId() : HGCalDetId() 
{
}

HGCTriggerHexDetId::HGCTriggerHexDetId(uint32_t rawid) : HGCalDetId(rawid) 
{
}

HGCTriggerHexDetId::
HGCTriggerHexDetId(ForwardSubdetector subdet, int zp, int lay, int wafertype, int wafer, int cell) : HGCalDetId(subdet, zp, lay, wafertype, wafer, cell) 
{  
}

HGCTriggerHexDetId::
HGCTriggerHexDetId(const DetId& gen) : HGCalDetId(gen)
{
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



