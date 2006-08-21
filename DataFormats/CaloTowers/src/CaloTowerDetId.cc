#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "FWCore/Utilities/interface/Exception.h"

static const int EncodingVersion = 1;

CaloTowerDetId::CaloTowerDetId() : DetId() {
}
  
CaloTowerDetId::CaloTowerDetId(uint32_t rawid) : DetId(rawid) {
  if (encodingVersion()==0) {
     id_=(id_&0xFFFFFF80u)|(iphi())|((EncodingVersion&0x3)<<16);
  }
}
  
CaloTowerDetId::CaloTowerDetId(int ieta, int iphi) : DetId(Calo,SubdetId) {
  id_|= 
    ((EncodingVersion&0x3)<<16) |
    ((ieta>0)?(0x2000|((ieta&0x3F)<<7)):(((-ieta)&0x3f)<<7)) |
    (iphi&0x7F);
}
  
CaloTowerDetId::CaloTowerDetId(const DetId& gen) {
  if (!gen.null() && (gen.det()!=Calo || gen.subdetId()!=SubdetId)) {
    throw cms::Exception("Invalid DetId") << "Cannot initialize CaloTowerDetId from " << std::hex << gen.rawId() << std::dec;
  }
  id_=gen.rawId(); 
  if (encodingVersion()==0) {
    id_=(id_&0xFFFFFF80u)|(iphi())|((EncodingVersion&0x3)<<16);
  }
}
  
CaloTowerDetId& CaloTowerDetId::operator=(const DetId& gen) {
  if (!gen.null() && (gen.det()!=Calo || gen.subdetId()!=SubdetId)) {
    throw cms::Exception("Invalid DetId") << "Cannot assign CaloTowerDetId from " << std::hex << gen.rawId() << std::dec;
  }
  id_=gen.rawId();
  if (encodingVersion()==0) {
    id_=(id_&0xFFFFFF80u)|(iphi())|((EncodingVersion&0x3)<<16);
  }
  return *this;
}

int CaloTowerDetId::iphi() const {
  int retval=id_&0x7F;
  if (encodingVersion()==0 && ietaAbs()<40) 
    retval=((retval+1)%72)+1;
  return retval;
}  

std::ostream& operator<<(std::ostream& s, const CaloTowerDetId& id) {
  return s << "Tower (" << id.ieta() << "," << id.iphi() << ")";
}
