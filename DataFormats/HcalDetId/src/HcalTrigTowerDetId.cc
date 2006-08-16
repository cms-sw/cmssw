#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"

const HcalTrigTowerDetId HcalTrigTowerDetId::Undefined(0x4a000000u);

HcalTrigTowerDetId::HcalTrigTowerDetId() {
}


HcalTrigTowerDetId::HcalTrigTowerDetId(uint32_t rawid) : DetId(rawid) {
}

HcalTrigTowerDetId::HcalTrigTowerDetId(int ieta, int iphi) : DetId(Hcal,HcalTriggerTower) {
  id_|=((ieta>0)?(0x2000|(ieta<<7)):((-ieta)<<7)) |
    (iphi&0x7F);
}

HcalTrigTowerDetId::HcalTrigTowerDetId(const DetId& gen) {
  if (!gen.null() && (gen.det()!=Hcal || gen.subdetId()!=HcalTriggerTower)) {
    throw new std::exception();
  }
  id_=gen.rawId();
}

HcalTrigTowerDetId& HcalTrigTowerDetId::operator=(const DetId& gen) {
  if (!gen.null() && (gen.det()!=Hcal || gen.subdetId()!=HcalTriggerTower)) {
    throw new std::exception();
  }
  id_=gen.rawId();
  return *this;
}

std::ostream& operator<<(std::ostream& s,const HcalTrigTowerDetId& id) {
  return s << "(HcalTrigTower " << id.ieta() << ',' << id.iphi() << ')';
}


