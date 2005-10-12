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

//New SM numbering scheme. Avoids discontinuity in phi crossing \eta=0  
int EBDetId::ism() const {
  int id = ( iphi() - 1 ) / 20 + 1;
  if ( zside() < 0 ) id += 18;
  return id;
}

//Following TB 2004  numbering scheme 
int EBDetId::ic() const {
  int ie = ieta() -1;
  int ip = ( iphi() -1 ) % 20;
  if ( ie < MIN_IETA - 1 || ie >= MAX_IETA || 
       ip < MIN_IPHI - 1 || ip >= MAX_IPHI ) return -1;

  return ie * kChannelsPerCard * kTowersInPhi + ip + 1;
}

//Maintains SM crystals in bunch of 1700 indices
int EBDetId::hashedIndex() const {
  return (ism()-1) * 1700 + ic() -1;
}
  
std::ostream& operator<<(std::ostream& s,const EBDetId& id) {
  return s << "(EB " << id.ieta() << ',' << id.iphi() << ')';
}
  
