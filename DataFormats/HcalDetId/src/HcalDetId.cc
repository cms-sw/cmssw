#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <ostream>
#include <iostream>

const HcalDetId HcalDetId::Undefined(HcalEmpty,0,0,0);

HcalDetId::HcalDetId() : DetId() {
}

HcalDetId::HcalDetId(uint32_t rawid) : DetId(newForm(rawid)) {
}

HcalDetId::HcalDetId(HcalSubdetector subdet, int tower_ieta, int tower_iphi, int depth) : DetId(Hcal,subdet) {
  // (no checking at this point!)
  id_ |= (kHcalIdFormat2) | ((depth&kHcalDepthMask2)<<kHcalDepthOffset2) |
    ((tower_ieta>0)?(kHcalZsideMask2|(tower_ieta<<kHcalEtaOffset2)):((-tower_ieta)<<kHcalEtaOffset2)) |
    (tower_iphi&kHcalPhiMask2);
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
    id_ = newForm(gen.rawId());
  } else {
    id_ = gen.rawId();
  }
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
    id_ = newForm(gen.rawId());
  } else {
    id_ = gen.rawId();
  }
  return (*this);
}
 
bool HcalDetId::operator==(DetId gen) const {
  uint32_t rawid = gen.rawId();
  if (rawid == id_) return true;
  int zsid, eta, phi, dep;
  unpackId(rawid, zsid, eta, phi, dep);
  bool result = (((id_&kHcalIdMask) == (rawid&kHcalIdMask)) && (zsid==zside())
		 && (eta==ietaAbs()) && (phi==iphi()) && (dep==depth()));
  return result;
}

bool HcalDetId::operator!=(DetId gen) const {
  uint32_t rawid = gen.rawId();
  if (rawid == id_) return false;
  int zsid, eta, phi, dep;
  unpackId(rawid, zsid, eta, phi, dep);
  bool result = (((id_&kHcalIdMask)!=(rawid&kHcalIdMask)) || (zsid!=zside())
		 || (eta!=ietaAbs()) || (phi!=iphi()) || (dep!=depth()));
  return result;
}

bool HcalDetId::operator<(DetId gen) const {
  uint32_t rawid = gen.rawId();
  if ((rawid&kHcalIdFormat2)==(id_&kHcalIdFormat2)) {
    return id_<rawid;
  } else {
    int zsid, eta, phi, dep;
    unpackId(rawid, zsid, eta, phi, dep);
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
  int dep = depth();
  if (subdet() == HcalForward) {
    if (dep > 2) dep -= 2;
  }
  return dep;
}

uint32_t HcalDetId::maskDepth() const {
  if (oldFormat())  return (id_|kHcalDepthSet1);
  else              return (id_|kHcalDepthSet2);
}

uint32_t HcalDetId::otherForm() const {
  uint32_t rawid = (id_&kHcalIdMask);
  if (oldFormat()) {
    rawid = newForm(id_);
  } else {
    rawid |= ((depth()&kHcalDepthMask1)<<kHcalDepthOffset1) |
      ((ieta()>0)?(kHcalZsideMask1|(ieta()<<kHcalEtaOffset1)):((-ieta())<<kHcalEtaOffset1)) |
      (iphi()&kHcalPhiMask1);
  }
  return rawid;
}

void HcalDetId::changeForm() {
  id_ = otherForm();
}

uint32_t HcalDetId::newForm() const {
  return newForm(id_);
}

uint32_t HcalDetId::newForm(const uint32_t& inpid) {
  uint32_t rawid(inpid);
  if ((rawid&kHcalIdFormat2)==0) {
    int zsid, eta, phi, dep;
    unpackId(rawid, zsid, eta, phi, dep);
    rawid    = inpid&kHcalIdMask;
    rawid   |= (kHcalIdFormat2) | ((dep&kHcalDepthMask2)<<kHcalDepthOffset2) |
      ((zsid>0)?(kHcalZsideMask2|(eta<<kHcalEtaOffset2)):((eta)<<kHcalEtaOffset2)) |
      (phi&kHcalPhiMask2);
  }
  return rawid;
}
 
bool HcalDetId::sameBaseDetId(const DetId& gen) const {
  uint32_t rawid = gen.rawId();
  if (rawid == id_) return true;
  int zsid, eta, phi, dep;
  if ((id_&kHcalIdMask) != (rawid&kHcalIdMask)) return false;
  unpackId(rawid, zsid, eta, phi, dep);
  if (subdet() == HcalForward && dep > 2) dep -= 2;
  bool result = ((zsid==zside()) && (eta==ietaAbs()) && (phi==iphi()) && 
		 (dep==hfdepth()));
  return result;
}

HcalDetId HcalDetId::baseDetId() const {
  if (subdet() != HcalForward || depth() <= 2) {
    return HcalDetId(id_);
  } else {
    int zsid, eta, phi, dep;
    unpackId(id_, zsid, eta, phi, dep);
    dep     -= 2;
    uint32_t rawid    = id_&kHcalIdMask;
    rawid   |= (kHcalIdFormat2) | ((dep&kHcalDepthMask2)<<kHcalDepthOffset2) |
      ((zsid>0)?(kHcalZsideMask2|(eta<<kHcalEtaOffset2)):((eta)<<kHcalEtaOffset2)) |
      (phi&kHcalPhiMask2);
    return HcalDetId(rawid);
  }
}

HcalDetId HcalDetId::secondAnodeId() const {
  if (subdet() != HcalForward || depth() > 2) {
    return HcalDetId(id_);
  } else {
    int zsid, eta, phi, dep;
    unpackId(id_, zsid, eta, phi, dep);
    dep     += 2;
    uint32_t rawid    = id_&kHcalIdMask;
    rawid   |= (kHcalIdFormat2) | ((dep&kHcalDepthMask2)<<kHcalDepthOffset2) |
      ((zsid>0)?(kHcalZsideMask2|(eta<<kHcalEtaOffset2)):((eta)<<kHcalEtaOffset2)) |
      (phi&kHcalPhiMask2);
    return HcalDetId(rawid);
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

void HcalDetId::newFromOld(const uint32_t& rawid) {
  id_ = newForm(rawid);
}

void HcalDetId::unpackId(const uint32_t& rawid, int& zsid, int& eta, int& phi,
			 int& dep) {
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


