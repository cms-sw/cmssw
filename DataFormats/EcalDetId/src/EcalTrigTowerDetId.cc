#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"



EcalTrigTowerDetId::EcalTrigTowerDetId() {
}
  
  
EcalTrigTowerDetId::EcalTrigTowerDetId(uint32_t rawid) : DetId(rawid) {
}
  
EcalTrigTowerDetId::EcalTrigTowerDetId(int ieta, int iphi) : DetId(Ecal,EcalTriggerTower) {
  id_|=((ieta>0)?(0x2000|(ieta<<7)):((-ieta)<<7)) |
    (iphi&0x7F);
}
  
EcalTrigTowerDetId::EcalTrigTowerDetId(const DetId& gen) {
  if (gen.det()!=Ecal || gen.subdetId()!=EcalTriggerTower) {
    throw new std::exception();
  }
  id_=gen.rawId();
}
  
EcalTrigTowerDetId& EcalTrigTowerDetId::operator=(const DetId& gen) {
  if (gen.det()!=Ecal || gen.subdetId()!=EcalTriggerTower) {
    throw new std::exception();
  }
  id_=gen.rawId();
  return *this;
}
  
std::ostream& operator<<(std::ostream& s,const EcalTrigTowerDetId& id) {
  return s << "(EcalTrigTower " << id.ieta() << ',' << id.iphi() << ')';
}

