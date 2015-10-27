#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>

CaloTowerDetId::CaloTowerDetId() : DetId() {
}
  
CaloTowerDetId::CaloTowerDetId(uint32_t rawid) : DetId(rawid&0xFFF0FFFFu) {
  
}
  
CaloTowerDetId::CaloTowerDetId(int ieta, int iphi) : DetId(Calo,SubdetId) {
  id_|= 
    ((ieta>0)?(0x2000|((ieta&0x3F)<<7)):(((-ieta)&0x3f)<<7)) |
    (iphi&0x7F);
}
  
CaloTowerDetId::CaloTowerDetId(const DetId& gen) {
  if (!gen.null() && (gen.det()!=Calo || gen.subdetId()!=SubdetId)) {
    throw cms::Exception("Invalid DetId") << "Cannot initialize CaloTowerDetId from " << std::hex << gen.rawId() << std::dec;
  }
  id_=gen.rawId(); 
}
  
CaloTowerDetId& CaloTowerDetId::operator=(const DetId& gen) {
  if (!gen.null() && (gen.det()!=Calo || gen.subdetId()!=SubdetId)) {
    throw cms::Exception("Invalid DetId") << "Cannot assign CaloTowerDetId from " << std::hex << gen.rawId() << std::dec;
  }
  id_=gen.rawId();
  return *this;
}

int CaloTowerDetId::iphi() const {
  int retval=id_&0x7F;
  return retval;
}

std::ostream& operator<<(std::ostream& s, const CaloTowerDetId& id) {
  return s << "Tower (" << id.ieta() << "," << id.iphi() << ")";
}
