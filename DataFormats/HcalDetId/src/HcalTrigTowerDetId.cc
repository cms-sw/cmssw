#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

const HcalTrigTowerDetId HcalTrigTowerDetId::Undefined(0x4a000000u);

HcalTrigTowerDetId::HcalTrigTowerDetId() {
}


HcalTrigTowerDetId::HcalTrigTowerDetId(uint32_t rawid) : DetId(rawid) {
}

HcalTrigTowerDetId::HcalTrigTowerDetId(int ieta, int iphi) : DetId(Hcal,HcalTriggerTower) {
  id_|=((ieta>0)?(kHcalZsideMask|(ieta<<kHcalEtaOffset)):((-ieta)<<kHcalEtaOffset)) |
    (iphi&kHcalPhiMask);
// Default to depth = 0 & version = 0
}

HcalTrigTowerDetId::HcalTrigTowerDetId(int ieta, int iphi, int depth) : DetId(Hcal,HcalTriggerTower) {
  const int ones = depth % 10;
  const int tens = (depth - ones) / 10;
  // version convension   0 : default for 3x2 TP; 1, 2, 3 : for future & currently version = 1 is for 1x1 TP
  // Note that in this conversion, depth can take values from 0 to 9 for different purpose 
  // -> so depth should be = ones!
  id_|=((ones&kHcalDepthMask)<<kHcalDepthOffset) |
    ((ieta>0)?(kHcalZsideMask|(ieta<<kHcalEtaOffset)):((-ieta)<<kHcalEtaOffset)) |
    (iphi&kHcalPhiMask);

  const int version = tens;
  if( version > 9 ){ // do NOT envision to have versions over 9...  
     edm::LogError("HcalTrigTowerDetId")<<"in its ctor using depth, version larger than 9 (too many of it!)?"<<std::endl;
  }

  id_|=((version&kHcalVersMask)<<kHcalVersOffset);
}

HcalTrigTowerDetId::HcalTrigTowerDetId(int ieta, int iphi, int depth, int version) : DetId(Hcal,HcalTriggerTower) {
  id_|=((depth&kHcalDepthMask)<<kHcalDepthOffset) |
    ((ieta>0)?(kHcalZsideMask|(ieta<<kHcalEtaOffset)):((-ieta)<<kHcalEtaOffset)) |
    (iphi&kHcalPhiMask);
  id_|=((version&kHcalVersMask)<<kHcalVersOffset);
}
 
HcalTrigTowerDetId::HcalTrigTowerDetId(const DetId& gen) {
  if (!gen.null() && (gen.det()!=Hcal || gen.subdetId()!=HcalTriggerTower)) {
    throw cms::Exception("Invalid DetId") << "Cannot initialize HcalTrigTowerDetId from " << std::hex << gen.rawId() << std::dec; 
  }
  id_=gen.rawId();
}

void HcalTrigTowerDetId::setVersion(int version) {
  id_|=((version&kHcalVersMask)<<kHcalVersOffset);
}

HcalTrigTowerDetId& HcalTrigTowerDetId::operator=(const DetId& gen) {
  if (!gen.null() && (gen.det()!=Hcal || gen.subdetId()!=HcalTriggerTower)) {
    throw cms::Exception("Invalid DetId") << "Cannot assign HcalTrigTowerDetId from " << std::hex << gen.rawId() << std::dec; 
  }
  id_=gen.rawId();
  return *this;
}

std::ostream& operator<<(std::ostream& s,const HcalTrigTowerDetId& id) {
  s << "(HcalTrigTower v" << id.version() << ": " << id.ieta() << ',' << id.iphi();
  if (id.depth()>0) s << ',' << id.depth();
  return s << ')';
}


