#include "DataFormats/EcalDetId/interface/EBDetId.h"



EBDetId::EBDetId() : DetId() {
}
  
EBDetId::EBDetId(uint32_t rawid) : DetId(rawid) {
}
  
EBDetId::EBDetId(int crystal_ieta, int crystal_iphi) : DetId(Ecal,EcalBarrel) {
  // (no checking at this point!)
  id_|=((crystal_ieta>0)?(0x10000|(crystal_ieta<<9)):((-crystal_ieta)<<9))|(crystal_iphi&0x1FF);
}
  
EBDetId::EBDetId(const DetId& gen) {
  if (gen.det()!=Ecal || gen.subdetId()!=EcalBarrel) {
    throw new std::exception();
  }
  id_=gen.rawId();
}
  
EBDetId& EBDetId::operator=(const DetId& gen) {
  if (gen.det()!=Ecal || gen.subdetId()!=EcalBarrel) {
    throw new std::exception();
  }
  id_=gen.rawId();
  return *this;
}
  
int EBDetId::hashedIndex() const {
  return (iphi()-1)*MAX_IETA*2+(ietaAbs()-1)+((zside()>0)?(MAX_IETA):(0));
}
  
std::ostream& operator<<(std::ostream& s,const EBDetId& id) {
  return s << "(EB " << id.ieta() << ',' << id.iphi() << ')';
}
  
