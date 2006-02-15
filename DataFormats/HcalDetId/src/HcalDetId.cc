#include "DataFormats/HcalDetId/interface/HcalDetId.h"

const HcalDetId HcalDetId::Undefined(HcalEmpty,0,0,0);

HcalDetId::HcalDetId() : DetId() {
}

HcalDetId::HcalDetId(uint32_t rawid) : DetId(rawid) {
}

HcalDetId::HcalDetId(HcalSubdetector subdet, int tower_ieta, int tower_iphi, int depth) : DetId(Hcal,subdet) {
  // (no checking at this point!)
  id_ |= ((depth&0x7)<<14) |
    ((tower_ieta>0)?(0x2000|(tower_ieta<<7)):((-tower_ieta)<<7)) |
    (tower_iphi&0x7F);
}

HcalDetId::HcalDetId(const DetId& gen) {
  HcalSubdetector subdet=(HcalSubdetector(gen.subdetId()));
  if (gen.det()!=Hcal || 
      (subdet!=HcalBarrel && subdet!=HcalEndcap && 
       subdet!=HcalOuter && subdet!=HcalForward ))
    {
      throw new std::exception();
    }  id_=gen.rawId();
}

HcalDetId& HcalDetId::operator=(const DetId& gen) {
  HcalSubdetector subdet=(HcalSubdetector(gen.subdetId()));
  if (gen.det()!=Hcal || 
      (subdet!=HcalBarrel && subdet!=HcalEndcap && 
       subdet!=HcalOuter && subdet!=HcalForward ))
    {
      throw new std::exception();
    }  id_=gen.rawId();
  return (*this);
}

std::ostream& operator<<(std::ostream& s,const HcalDetId& id) {
  switch (id.subdet()) {
  case(HcalBarrel) : return s << "(HB " << id.ieta() << ',' << id.iphi() << ',' << id.depth() << ')';
  case(HcalEndcap) : return s << "(HE " << id.ieta() << ',' << id.iphi() << ',' << id.depth() << ')';
  case(HcalForward) : return s << "(HF " << id.ieta() << ',' << id.iphi() << ',' << id.depth() << ')';
  case(HcalOuter) : return s << "(HO " << id.ieta() << ',' << id.iphi() << ')';
  default : return s << id.rawId();
  }
}

