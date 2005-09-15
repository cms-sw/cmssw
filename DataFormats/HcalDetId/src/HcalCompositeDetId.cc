#include "DataFormats/HcalDetId/interface/HcalCompositeDetId.h"

namespace cms {

  HcalCompositeDetId::HcalCompositeDetId() {
  }
  

  HcalCompositeDetId::HcalCompositeDetId(uint32_t rawid) : DetId(rawid) {
  }
  
  HcalCompositeDetId::HcalCompositeDetId(CompositeType composite_type, int composite_data, int ieta, int iphi) : DetId(Hcal,HcalComposite) {
    id_|= (int(composite_type&0xF)<<21) |
      ((composite_data&0x7F)<<14) |
      ((ieta>0)?(0x2000|((ieta&0x3F)<<7)):(((-ieta)&0x3f)<<7)) |
      (iphi&0x7F);
  }
  
  HcalCompositeDetId::HcalCompositeDetId(const DetId& gen) {
    if (gen.det()!=Hcal || gen.subdetId()!=HcalComposite) {
      throw new std::exception();
    }
    id_=gen.rawId();    
  }
  
  HcalCompositeDetId& HcalCompositeDetId::operator=(const DetId& gen) {
    if (gen.det()!=Hcal || gen.subdetId()!=HcalComposite) {
      throw new std::exception();
    }
    id_=gen.rawId();
    return *this;
  }
  

}
