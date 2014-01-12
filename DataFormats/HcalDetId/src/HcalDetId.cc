#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <ostream>

const HcalDetId HcalDetId::Undefined(HcalEmpty,0,0,0);

HcalDetId::HcalDetId() : DetId() {
}

HcalDetId::HcalDetId(uint32_t rawid) : DetId(rawid) {
}

HcalDetId::HcalDetId(HcalSubdetector subdet, int tower_ieta, int tower_iphi, int depth) : DetId(Hcal,subdet) {
  // (no checking at this point!)
  id_ |= (0x1000000) | ((depth&0xF)<<19) |
    ((tower_ieta>0)?(0x800000|((tower_ieta&0x1FF)<<10)):(((-tower_ieta)&0x1FF)<<10)) |
    (tower_iphi&0x3FF);
}

HcalDetId::HcalDetId(HcalSubdetector subdet, int tower_ieta, int tower_iphi, int depth, bool oldFormat) : DetId(Hcal,subdet) {
  // (no checking at this point!)
  if (oldFormat) {
    id_ |= ((depth&0x1F)<<14) |
      ((tower_ieta>0)?(0x2000|((tower_ieta&0x3F)<<7)):(((-tower_ieta)&0x3F)<<7)) |
      (tower_iphi&0x7F);
  } else {
    id_ |= (0x1000000) | ((depth&0xF)<<19) |
      ((tower_ieta>0)?(0x800000|((tower_ieta&0x1FF)<<10)):(((-tower_ieta)&0x1FF)<<10)) |
      (tower_iphi&0x3FF);
  }
}

HcalDetId::HcalDetId(const DetId& gen) {
  if (!gen.null()) {
    HcalSubdetector subdet=(HcalSubdetector(gen.subdetId()));
    if (gen.det()!=Hcal || 
	(subdet!=HcalBarrel && subdet!=HcalEndcap && 
	 subdet!=HcalOuter && subdet!=HcalForward )) {
      throw cms::Exception("Invalid DetId") << "Cannot initialize HcalDetId from " << std::hex << gen.rawId() << std::dec; 
    }  
  }
  id_=gen.rawId();
}

HcalDetId& HcalDetId::operator=(const DetId& gen) {
  if (!gen.null()) {
    HcalSubdetector subdet=(HcalSubdetector(gen.subdetId()));
    if (gen.det()!=Hcal || 
	(subdet!=HcalBarrel && subdet!=HcalEndcap && 
	 subdet!=HcalOuter && subdet!=HcalForward ))
      {
	throw cms::Exception("Invalid DetId") << "Cannot assign HcalDetId from " << std::hex << gen.rawId() << std::dec; 
      }  
  }
  id_=gen.rawId();
  return (*this);
}

int HcalDetId::zside() const {
  if (oldFormat()) return (id_&0x2000)?(1):(-1);
  else             return (id_&0x800000)?(1):(-1);
}

int HcalDetId::ietaAbs() const { 
  if (oldFormat()) return (id_>>7)&0x3F; 
  else             return (id_>>10)&0x1FF;
}
  
int HcalDetId::iphi() const { 
  if (oldFormat()) return id_&0x7F; 
  else             return id_&0x3FF;
}

int HcalDetId::depth() const {
  if (oldFormat())  return (id_>>14)&0x1F;
  else              return (id_>>19)&0xF;
}

int HcalDetId::crystal_iphi_low() const { 
  int simple_iphi=((iphi()-1)*5)+1; 
  simple_iphi+=10;
  return ((simple_iphi>360)?(simple_iphi-360):(simple_iphi));
}

int HcalDetId::crystal_iphi_high() const { 
  int simple_iphi=((iphi()-1)*5)+5; 
  simple_iphi+=10;
  return ((simple_iphi>360)?(simple_iphi-360):(simple_iphi));
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


