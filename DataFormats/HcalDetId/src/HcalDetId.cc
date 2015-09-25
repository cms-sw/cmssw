#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <ostream>
#include <iostream>

const HcalDetId HcalDetId::Undefined(HcalEmpty,0,0,0);

HcalDetId::HcalDetId() : DetId() {
}

HcalDetId::HcalDetId(uint32_t rawid) : DetId(rawid) {
}

HcalDetId::HcalDetId(HcalSubdetector subdet, int tower_ieta, int tower_iphi, int depth, bool oldFormat) : DetId(Hcal,subdet) {
  // (no checking at this point!)
  if (oldFormat) {
    id_ |= ((depth&kHcalDepthMask1)<<kHcalDepthOffset1) |
      ((tower_ieta>0)?(kHcalZsideMask1|(tower_ieta<<kHcalEtaOffset1)):((-tower_ieta)<<kHcalEtaOffset1)) |
      (tower_iphi&kHcalPhiMask1);
  } else {
    id_ |= (kHcalIdFormat2) | ((depth&kHcalDepthMask2)<<kHcalDepthOffset2) |
      ((tower_ieta>0)?(kHcalZsideMask2|(tower_ieta<<kHcalEtaOffset2)):((-tower_ieta)<<kHcalEtaOffset2)) |
      (tower_iphi&kHcalPhiMask2);
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
  if ((rawid&kHcalIdFormat2)==0) {
    zsid = (rawid&kHcalZsideMask1)?(1):(-1);
    eta  = (rawid>>kHcalEtaOffset1)&kHcalEtaMask1;
    phi  = rawid&kHcalPhiMask1;
    dep  = (rawid>>kHcalDepthOffset1)&kHcalDepthMask1;
  } else {
    zsid = (rawid&kHcalZsideMask2)?(1):(-1);
    eta  = (rawid>>kHcalEtaOffset2)&kHcalEtaMask2;
    phi  = rawid&kHcalPhiMask2;
    dep  = (rawid>>kHcalDepthOffset2)&kHcalDepthMask2;
  }
  bool result=(zsid==zside() && eta==ietaAbs() && phi==iphi() && dep==depth());
  return result;
}

bool HcalDetId::operator!=(DetId gen) const {
  uint32_t rawid = gen.rawId();
  if (rawid == id_) return false;
  int zsid, eta, phi, dep;
  if ((rawid&kHcalIdFormat2)==0) {
    zsid = (rawid&kHcalZsideMask1)?(1):(-1);
    eta  = (rawid>>kHcalEtaOffset1)&kHcalEtaMask1;
    phi  = rawid&kHcalPhiMask1;
    dep  = (rawid>>kHcalDepthOffset1)&kHcalDepthMask1;
  } else {
    zsid = (rawid&kHcalZsideMask2)?(1):(-1);
    eta  = (rawid>>kHcalEtaOffset2)&kHcalEtaMask2;
    phi  = rawid&kHcalPhiMask2;
    dep  = (rawid>>kHcalDepthOffset2)&kHcalDepthMask2;
  }
  bool result=(zsid!=zside() || eta!=ietaAbs() || phi!=iphi() || dep!=depth());
  return result;
}

bool HcalDetId::operator<(DetId gen) const {
  uint32_t rawid = gen.rawId();
  if ((rawid&kHcalIdFormat2)==(id_&kHcalIdFormat2)) {
    return id_<rawid;
  } else {
    int zsid, eta, phi, dep;
    if ((rawid&kHcalIdFormat2)==0) {
      zsid = (rawid&kHcalZsideMask1)?(1):(-1);
      eta  = (rawid>>kHcalEtaOffset1)&kHcalEtaMask1;
      phi  = rawid&kHcalPhiMask1;
      dep  = (rawid>>kHcalDepthOffset1)&kHcalDepthMask1;
    } else {
      zsid = (rawid&kHcalZsideMask2)?(1):(-1);
      eta  = (rawid>>kHcalEtaOffset2)&kHcalEtaMask2;
      phi  = rawid&kHcalPhiMask2;
      dep  = (rawid>>kHcalDepthOffset2)&kHcalDepthMask2;
    }
    rawid &= kHcalIdMask;
    if (oldFormat()) {
      rawid |= ((dep&kHcalDepthMask1)<<kHcalDepthOffset1) |
	((zsid>0)?(kHcalZsideMask1|(eta<<kHcalEtaOffset1)):((eta)<<kHcalEtaOffset1)) |
	(phi&kHcalPhiMask1);
    } else {
      rawid |= (kHcalIdFormat2) | ((dep&kHcalDepthMask2)<<kHcalDepthOffset2) |
	((zsid>0)?(kHcalZsideMask2|(eta<<kHcalEtaOffset2)):((eta)<<kHcalEtaOffset2)) |
	(phi&kHcalPhiMask2);
    }
    return (id_<rawid);
  }
}

int HcalDetId::zside() const {
  if (oldFormat()) return (id_&kHcalZsideMask1)?(1):(-1);
  else             return (id_&kHcalZsideMask2)?(1):(-1);
}

int HcalDetId::ietaAbs() const { 
  if (oldFormat()) return (id_>>kHcalEtaOffset1)&kHcalEtaMask1; 
  else             return (id_>>kHcalEtaOffset2)&kHcalEtaMask2;
}
  
int HcalDetId::iphi() const { 
  if (oldFormat()) return id_&kHcalPhiMask1; 
  else             return id_&kHcalPhiMask2;
}

int HcalDetId::depth() const {
  if (oldFormat())  return (id_>>kHcalDepthOffset1)&kHcalDepthMask1;
  else              return (id_>>kHcalDepthOffset2)&kHcalDepthMask2;
}

int HcalDetId::hfdepth() const {
  if (subdet() == HcalForward) {
    int dep = depth();
    if (dep > 2) dep -= 2;
    return dep;
  } else {
    return depth();
  }
}

uint32_t HcalDetId::maskDepth() const {
  if (oldFormat())  return (id_|kHcalDepthSet1);
  else              return (id_|kHcalDepthSet2);
}

uint32_t HcalDetId::otherForm() const {
  uint32_t rawId = (id_&kHcalIdMask);
  if (oldFormat()) {
    rawId |= (kHcalIdFormat2) | ((hfdepth()&kHcalDepthMask2)<<kHcalDepthOffset2) |
      ((ieta()>0)?(kHcalZsideMask2|(ieta()<<kHcalEtaOffset2)):((-ieta())<<kHcalEtaOffset2)) |
      (iphi()&kHcalPhiMask2);
  } else {
    rawId |= ((hfdepth()&kHcalDepthMask1)<<kHcalDepthOffset1) |
      ((ieta()>0)?(kHcalZsideMask1|(ieta()<<kHcalEtaOffset1)):((-ieta())<<kHcalEtaOffset1)) |
      (iphi()&kHcalPhiMask1);
  }
  return rawId;
}

uint32_t HcalDetId::newForm() const {
  if (oldFormat()) {
    uint32_t rawId = (id_&kHcalIdMask);
    rawId |= (kHcalIdFormat2) | ((hfdepth()&kHcalDepthMask2)<<kHcalDepthOffset2) |
      ((ieta()>0)?(kHcalZsideMask2|(ieta()<<kHcalEtaOffset2)):((-ieta())<<kHcalEtaOffset2)) |
      (iphi()&kHcalPhiMask2);
    return rawId;
  } else {
    return id_;
  }
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
  case(HcalTriggerTower) : return s << "(HT " << id.ieta() << ',' << id.iphi() << ')';
  default : return s << id.rawId();
  }
}


