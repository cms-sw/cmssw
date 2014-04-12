#include "DataFormats/HcalDetId/interface/HcalOtherDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>
HcalOtherDetId::HcalOtherDetId() {
}
  
HcalOtherDetId::HcalOtherDetId(uint32_t rawid) : DetId(rawid) {
}
  
HcalOtherDetId::HcalOtherDetId(HcalOtherSubdetector other_type)
  : DetId(Hcal,HcalOther) {
  id_|= (int(other_type&0x1F)<<20);
}
  
HcalOtherDetId::HcalOtherDetId(const DetId& gen) {
  if (gen.det()!=Hcal || gen.subdetId()!=HcalOther) {
    throw cms::Exception("Invalid DetId") << "Cannot initialize HcalOtherDetId from " << std::hex << gen.rawId() << std::dec;
  }
  id_=gen.rawId();    
}
  
HcalOtherDetId& HcalOtherDetId::operator=(const DetId& gen) {
  if (gen.det()!=Hcal || gen.subdetId()!=HcalOther) {
    throw cms::Exception("Invalid DetId") << "Cannot assign HcalOtherDetId from " << std::hex << gen.rawId() << std::dec; 
  }
  id_=gen.rawId();
  return *this;
}
  


