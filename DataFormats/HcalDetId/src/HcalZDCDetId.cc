#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"

HcalZDCDetId::HcalZDCDetId() : HcalOtherDetId() {
}


HcalZDCDetId::HcalZDCDetId(uint32_t rawid) : HcalOtherDetId(rawid) {
}

HcalZDCDetId::HcalZDCDetId(Section section, bool true_for_positive_eta, int depth) : HcalOtherDetId(HcalZDC) {
  id_|=(section&0x3)<<4;
  if (true_for_positive_eta) id_|=0x40;
  id_|=depth&0xF;
}

HcalZDCDetId::HcalZDCDetId(const DetId& gen) {
  if (!gen.null() && (gen.det()!=Hcal || gen.subdetId()!=HcalOther)) {
    throw new std::exception();
  }
  id_=gen.rawId();
  if (subdet()!=HcalZDC) {
    throw new std::exception();
  }
}

HcalZDCDetId& HcalZDCDetId::operator=(const DetId& gen) {
  if (!gen.null() && (gen.det()!=Hcal || gen.subdetId()!=HcalOther)) {
    throw new std::exception();
  }
  id_=gen.rawId();
  if (subdet()!=HcalZDC) {
    throw new std::exception();
  }
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

