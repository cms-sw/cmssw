#include "DataFormats/HcalDetId/interface/HcalCastorDetId.h"
#include "FWCore/Utilities/interface/Exception.h"

HcalCastorDetId::HcalCastorDetId() : DetId() {
}

HcalCastorDetId::HcalCastorDetId(uint32_t rawid) : DetId(rawid) {
}

HcalCastorDetId::HcalCastorDetId(Section section, bool true_for_positive_eta, int sector, 
int module) : 
DetId(DetId::Calo,SubdetectorId) {
  id_|=(section&0x3)<<7;
  id_|=(sector&0x3)<<6;
  if (true_for_positive_eta) id_|=0x40;
  id_|=module&0x3;
}

HcalCastorDetId::HcalCastorDetId(const DetId& gen) {
  if (!gen.null() && (gen.det()!=Calo || gen.subdetId()!=SubdetectorId)) {
    throw cms::Exception("Invalid DetId") << "Cannot initialize CASTORDetId from " << std::hex << gen.rawId() << std::dec; 
  }
  id_=gen.rawId();
}

HcalCastorDetId& HcalCastorDetId::operator=(const DetId& gen) {
  if (!gen.null() && (gen.det()!=Calo || gen.subdetId()!=SubdetectorId)) {
    throw cms::Exception("Invalid DetId") << "Cannot assign Castor DetId from " << std::hex << gen.rawId() << std::dec; 
  }
  id_=gen.rawId();
  return *this;
}

/*
int HcalCastorDetId::channel() const {
  int channelid = 16*(sector-1)+module;
  return channelid;
}
*/

std::ostream& operator<<(std::ostream& s,const HcalCastorDetId& id) {
  s << "(CASTOR" << ((id.zside()==1)?("+"):("-"));
  switch (id.section()) {
  case(HcalCastorDetId::EM) : s << " EM "; break;
  case(HcalCastorDetId::HAD) : s << " HAD "; break;
  default : s <<" UNKNOWN ";
  }
  return s << id.sector() << ',' << id.module() << ',' << ')';
}

