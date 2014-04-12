#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "FWCore/Utilities/interface/Exception.h"

const HcalTrigTowerDetId HcalTrigTowerDetId::Undefined(0x4a000000u);

HcalTrigTowerDetId::HcalTrigTowerDetId() {
}


HcalTrigTowerDetId::HcalTrigTowerDetId(uint32_t rawid) : DetId(rawid) {
}

HcalTrigTowerDetId::HcalTrigTowerDetId(int ieta, int iphi) : DetId(Hcal,HcalTriggerTower) {
  id_|=((ieta>0)?(0x2000|(ieta<<7)):((-ieta)<<7)) |
    (iphi&0x7F);
}

HcalTrigTowerDetId::HcalTrigTowerDetId(int ieta, int iphi, int depth) : DetId(Hcal,HcalTriggerTower) {
  id_|=((depth&0x7)<<14) |
    ((ieta>0)?(0x2000|(ieta<<7)):((-ieta)<<7)) |
    (iphi&0x7F);
}

HcalTrigTowerDetId::HcalTrigTowerDetId(const DetId& gen) {
  if (!gen.null() && (gen.det()!=Hcal || gen.subdetId()!=HcalTriggerTower)) {
    throw cms::Exception("Invalid DetId") << "Cannot initialize HcalTrigTowerDetId from " << std::hex << gen.rawId() << std::dec; 
  }
  id_=gen.rawId();
}

HcalTrigTowerDetId& HcalTrigTowerDetId::operator=(const DetId& gen) {
  if (!gen.null() && (gen.det()!=Hcal || gen.subdetId()!=HcalTriggerTower)) {
    throw cms::Exception("Invalid DetId") << "Cannot assign HcalTrigTowerDetId from " << std::hex << gen.rawId() << std::dec; 
  }
  id_=gen.rawId();
  return *this;
}

std::ostream& operator<<(std::ostream& s,const HcalTrigTowerDetId& id) {
  s << "(HcalTrigTower " << id.ieta() << ',' << id.iphi();
  if (id.depth()>0) s << ',' << id.depth();
  
  return s << ')';
}


