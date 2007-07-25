#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "FWCore/Utilities/interface/Exception.h"

HcalZDCDetId::HcalZDCDetId() : DetId() {
}


HcalZDCDetId::HcalZDCDetId(uint32_t rawid) : DetId(rawid) {
}

HcalZDCDetId::HcalZDCDetId(Section section, bool true_for_positive_eta, int depth) : DetId(DetId::Calo,SubdetectorId) {
  id_|=(section&0x3)<<4;
  if (true_for_positive_eta) id_|=0x40;
  id_|=depth&0xF;
}

HcalZDCDetId::HcalZDCDetId(const DetId& gen) {
  if (!gen.null() && (gen.det()!=Calo || gen.subdetId()!=SubdetectorId)) {
    throw cms::Exception("Invalid DetId") << "Cannot initialize ZDCDetId from " << std::hex << gen.rawId() << std::dec; 
  }
  id_=gen.rawId();
}

HcalZDCDetId& HcalZDCDetId::operator=(const DetId& gen) {
  if (!gen.null() && (gen.det()!=Calo || gen.subdetId()!=SubdetectorId)) {
    throw cms::Exception("Invalid DetId") << "Cannot assign ZDCDetId from " << std::hex << gen.rawId() << std::dec; 
  }
  id_=gen.rawId();
  return *this;
}


std::ostream& operator<<(std::ostream& s,const HcalZDCDetId& id) {
  s << "(ZDC" << ((id.zside()==1)?("+"):("-"));
  switch (id.section()) {
  case(HcalZDCDetId::EM) : s << " EM "; break;
  case(HcalZDCDetId::HAD) : s << " HAD "; break;
  case(HcalZDCDetId::LUM) : s << " LUM "; break;
  default : s <<" UNKNOWN ";
  }
  return s << id.depth() << ')';
}

