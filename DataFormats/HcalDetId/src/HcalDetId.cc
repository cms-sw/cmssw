#include "DataFormats/HcalDetId/interface/HcalDetId.h"

const HcalDetId HcalDetId::Undefined(HcalEmpty,0,0,0);

static const int EncodingVersion = 1;

HcalDetId::HcalDetId() : DetId() {
}

HcalDetId::HcalDetId(uint32_t rawid) : DetId(rawid) {
  if (!null() && encodingVersion()==0) {
    id_=(id_&0xFFFFFF80u)|(iphi())|((EncodingVersion&0x3)<<17);
  }
}

HcalDetId::HcalDetId(HcalSubdetector subdet, int tower_ieta, int tower_iphi, int depth) : DetId(Hcal,subdet) {
  // (no checking at this point!)
  id_ |=
    ((EncodingVersion&0x3)<<17) |
    ((depth&0x7)<<14) |
    ((tower_ieta>0)?(0x2000|(tower_ieta<<7)):((-tower_ieta)<<7)) |
    (tower_iphi&0x7F);
}

HcalDetId::HcalDetId(const DetId& gen) {
  if (!gen.null()) {
    HcalSubdetector subdet=(HcalSubdetector(gen.subdetId()));
    if (gen.det()!=Hcal || 
	(subdet!=HcalBarrel && subdet!=HcalEndcap && 
	 subdet!=HcalOuter && subdet!=HcalForward ))
      {
	throw new std::exception();
      }  
  }
  id_=gen.rawId();
  if (!null() && encodingVersion()==0) {
    id_=(id_&0xFFFFFF80u)|(iphi())|((EncodingVersion&0x3)<<17);
  }
}

HcalDetId& HcalDetId::operator=(const DetId& gen) {
  if (!gen.null()) {
    HcalSubdetector subdet=(HcalSubdetector(gen.subdetId()));
    if (gen.det()!=Hcal || 
	(subdet!=HcalBarrel && subdet!=HcalEndcap && 
	 subdet!=HcalOuter && subdet!=HcalForward ))
      {
	throw new std::exception();
      }  
  }
  id_=gen.rawId();
  if (!null() && encodingVersion()==0) {
    id_=(id_&0xFFFFFF80u)|(iphi())|((EncodingVersion&0x3)<<17);
  }
  return (*this);
}


HcalDetId::HcalDetId(const HcalDetId& gen) {
  id_=gen.rawId();
  if (!null() && encodingVersion()==0) {
    id_=(id_&0xFFFFFF80u)|(iphi())|((EncodingVersion&0x3)<<17);
  }
}

HcalDetId& HcalDetId::operator=(const HcalDetId& gen) {
  id_=gen.rawId();
  if (!null() && encodingVersion()==0) {
    id_=(id_&0xFFFFFF80u)|(iphi())|((EncodingVersion&0x3)<<17);
  }
  return (*this);
}

int HcalDetId::iphi() const {
  int retval=(id_&0x7F);
  if (!null() && encodingVersion()==0 && ietaAbs()<40) 
    retval=((retval+1)%72)+1;
  return retval;
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

