#include "DataFormats/HcalDetId/interface/HcalDetId.h"

namespace cms {

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

int HcalDetId::hashedIndex() const { 
  static const int NETA = 42;
  static const int NPHI = 72;
  static const int NDEPTH = 4;
  if (zside()>0) {
    return NETA*NPHI*NDEPTH+(depth()-1)+ietaAbs()*NDEPTH+iphi()*NDEPTH*NETA;
  } else {
    return (depth()-1)+ietaAbs()*NDEPTH+iphi()*NDEPTH*NETA;
  }
}

}

std::ostream& operator<<(std::ostream& s,const cms::HcalDetId& id) {
  switch (id.subdet()) {
  case(cms::HcalBarrel) : return s << "(HB " << id.ieta() << ',' << id.iphi() << ',' << id.depth() << ')';
  case(cms::HcalEndcap) : return s << "(HE " << id.ieta() << ',' << id.iphi() << ',' << id.depth() << ')';
  case(cms::HcalForward) : return s << "(HF " << id.ieta() << ',' << id.iphi() << ',' << id.depth() << ')';
  case(cms::HcalOuter) : return s << "(HO " << id.ieta() << ',' << id.iphi() << ')';
  default : return s << id.rawId();
  }
}
