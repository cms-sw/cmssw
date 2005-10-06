#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"

CaloTowerDetId::CaloTowerDetId() {
}
  
CaloTowerDetId::CaloTowerDetId(uint32_t rawid) : DetId(rawid) {
}
  
CaloTowerDetId::CaloTowerDetId(int ieta, int iphi) : DetId(Calo,SubdetId) {
  id_|= 
    ((ieta>0)?(0x2000|((ieta&0x3F)<<7)):(((-ieta)&0x3f)<<7)) |
    (iphi&0x7F);
}
  
CaloTowerDetId::CaloTowerDetId(const DetId& gen) {
  if (gen.det()!=Calo || gen.subdetId()!=SubdetId) {
    throw new std::exception();
  }
  id_=gen.rawId();    
}
  
CaloTowerDetId& CaloTowerDetId::operator=(const DetId& gen) {
  if (gen.det()!=Calo || gen.subdetId()!=SubdetId) {
    throw new std::exception();
  }
  id_=gen.rawId();
  return *this;
}
  

std::ostream& operator<<(std::ostream& s, const CaloTowerDetId& id) {
  return s << "Tower (" << id.ieta() << "," << id.iphi() << ")";
}
