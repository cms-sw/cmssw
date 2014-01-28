#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <iostream>

const HcalDetId HcalDetId::Undefined(HcalEmpty,0,0,0);

HcalDetId::HcalDetId() : DetId() {
}

HcalDetId::HcalDetId(uint32_t rawid) : DetId(rawid) {
}

HcalDetId::HcalDetId(HcalSubdetector subdet, int tower_ieta, int tower_iphi, int depth, bool oldFormat) : DetId(Hcal,subdet) {
  // (no checking at this point!)
  if (oldFormat) {
    id_ |= ((depth&0x1F)<<14) |
      ((tower_ieta>0)?(0x2000|((tower_ieta&0x3F)<<7)):(((-tower_ieta)&0x3F)<<7)) |
      (tower_iphi&0x7F);
  } else {
    id_ |= (0x1000000) | ((depth&0xF)<<20) |
      ((tower_ieta>0)?(0x80000|((tower_ieta&0x1FF)<<10)):(((-tower_ieta)&0x1FF)<<10)) |
      (tower_iphi&0x3FF);
  }
}

HcalDetId::HcalDetId(const DetId& gen) {
  if (!gen.null()) {
    HcalSubdetector subdet=(HcalSubdetector(gen.subdetId()));
    if (gen.det()!=Hcal || 
	(subdet!=HcalBarrel && subdet!=HcalEndcap && 
	 subdet!=HcalOuter && subdet!=HcalForward &&
	 subdet!=HcalTriggerTower && subdet!=HcalOther)) {
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
	 subdet!=HcalOuter && subdet!=HcalForward &&
	 subdet!=HcalTriggerTower && subdet!=HcalOther)) {
      throw cms::Exception("Invalid DetId") << "Cannot assign HcalDetId from " << std::hex << gen.rawId() << std::dec; 
    }  
  }
  id_=gen.rawId();
  return (*this);
}

bool HcalDetId::operator==(DetId gen) const {
  uint32_t rawid = gen.rawId();
  if (rawid == id_) return true;
  int zsid, eta, phi, dep;
  if ((rawid&0x1000000)==0) {
    zsid = (rawid&0x2000)?(1):(-1);
    eta  = (rawid>>7)&0x3F;
    phi  = rawid&0x7F;
    dep  = (rawid>>14)&0x1F;
  } else {
    zsid = (rawid&0x80000)?(1):(-1);
    eta  = (rawid>>10)&0x1FF;
    phi  = rawid&0x3FF;
    dep  = (rawid>>20)&0xF;
  }
  bool result=(zsid==zside() && eta==ietaAbs() && phi==iphi() && dep==depth());
  return result;
}

bool HcalDetId::operator!=(DetId gen) const {
  uint32_t rawid = gen.rawId();
  if (rawid == id_) return false;
  int zsid, eta, phi, dep;
  if ((rawid&0x1000000)==0) {
    zsid = (rawid&0x2000)?(1):(-1);
    eta  = (rawid>>7)&0x3F;
    phi  = rawid&0x7F;
    dep  = (rawid>>14)&0x1F;
  } else {
    zsid = (rawid&0x80000)?(1):(-1);
    eta  = (rawid>>10)&0x1FF;
    phi  = rawid&0x3FF;
    dep  = (rawid>>20)&0xF;
  }
  bool result=(zsid!=zside() || eta!=ietaAbs() || phi!=iphi() || dep!=depth());
  return result;
}

bool HcalDetId::operator<(DetId gen) const {
  uint32_t rawid = gen.rawId();
  if ((rawid&0x1000000)==(id_&0x1000000)) {
    return id_<rawid;
  } else {
    int zsid, eta, phi, dep;
    if ((rawid&0x1000000)==0) {
      zsid = (rawid&0x2000)?(1):(-1);
      eta  = (rawid>>7)&0x3F;
      phi  = rawid&0x7F;
      dep  = (rawid>>14)&0x1F;
    } else {
      zsid = (rawid&0x80000)?(1):(-1);
      eta  = (rawid>>10)&0x1FF;
      phi  = rawid&0x3FF;
      dep  = (rawid>>20)&0xF;
    }
    rawid = 0;
    if ((id_&0x1000000) == 0) {
      rawid |= ((dep&0x1F)<<14) |
	((zsid>0)?(0x2000|((eta&0x3F)<<7)):((eta&0x3F)<<7)) |
	(phi&0x7F);
    } else {
      rawid |= (0x1000000) | ((dep&0xF)<<20) |
	((zsid>0)?(0x80000|((eta&0x1FF)<<10)):((eta&0x1FF)<<10)) |
	(phi&0x3FF);
    }
    return (id_&0x1FFFFFF)<rawid;
  }
}

int HcalDetId::zside() const {
  if (oldFormat()) return (id_&0x2000)?(1):(-1);
  else             return (id_&0x80000)?(1):(-1);
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
  else              return (id_>>20)&0xF;
}

uint32_t HcalDetId::otherForm() const {
  uint32_t rawId = (id_&0xFE000000);
  if (oldFormat()) {
    rawId |= (0x1000000) | ((depth()&0xF)<<20) |
      ((ieta()>0)?(0x80000|((ieta()&0x1FF)<<10)):(((-ieta())&0x1FF)<<10)) |
      (iphi()&0x3FF);
  } else {
    rawId |= ((depth()&0x1F)<<14) |
      ((ieta()>0)?(0x2000|((ieta()&0x3F)<<7)):(((-ieta())&0x3F)<<7)) |
      (iphi()&0x7F);
  }
  return rawId;
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


