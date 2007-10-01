#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
using namespace reco;

HitPattern::HitPattern() {
  // init hit pattern array as 0x00000000, ..., 0x00000000
  for (int i=0; i<PatternSize; i++) hitPattern_[i] = 0; 
}

void HitPattern::set(const TrackingRecHit & hit, unsigned int i) {
  // ignore the rec hit if the number of hit is larger than the max
  if (i >= 32 * PatternSize / HitSize) return;

  // get rec hit det id and rec hit type
  DetId id = hit.geographicalId();
  uint32_t detid = id.det();
  uint32_t hitType = (uint32_t) hit.getType();

  // init pattern of this hit to 0
  uint32_t pattern = 0;

  // adding tracker/muon detector bit
  pattern += ((detid)&SubDetectorMask)<<SubDetectorOffset;
  
  // adding substructure (PXB,PXF,TIB,TID,TOB,TEC, or DT,CSC,RPC) bits 
  uint32_t subdet = id.subdetId();
  pattern += ((subdet)&SubstrMask)<<SubstrOffset;
  
  // adding layer/disk/wheel bits
  uint32_t layer = 0;
  if (detid == DetId::Tracker) {
    if (subdet == PixelSubdetector::PixelBarrel) 
      layer = PXBDetId(id).layer();
    else if (subdet == PixelSubdetector::PixelEndcap)
      layer = PXFDetId(id).disk();
    else if (subdet == StripSubdetector::TIB)
      layer = TIBDetId(id).layer();
    else if (subdet == StripSubdetector::TID)
      layer = TIDDetId(id).wheel();
    else if (subdet == StripSubdetector::TOB)
      layer = TOBDetId(id).layer();
    else if (subdet == StripSubdetector::TEC)
      layer = TECDetId(id).wheel();
  } else if (detid == DetId::Muon) {
    if (subdet == (uint32_t) MuonSubdetId::DT) 
      layer = DTLayerId(id.rawId()).layer();
    else if (subdet == (uint32_t) MuonSubdetId::CSC)
      layer = CSCDetId(id.rawId()).layer();
    else if (subdet == (uint32_t) MuonSubdetId::RPC)
      layer = RPCDetId(id.rawId()).layer();
  }
  pattern += (layer&LayerMask)<<LayerOffset;

  // adding hit type bits
  pattern += (hitType&HitTypeMask)<<HitTypeOffset;

  // set pattern for i-th hit
  setHitPattern(i, pattern);
}

void HitPattern::setHitPattern(int position, uint32_t pattern) {
  int offset = position * HitSize;
  for (int i=0; i<HitSize; i++) {
    int pos = offset + i;
    uint32_t bit = (pattern >> i) & 0x1;
    hitPattern_[pos / 32] += bit << ((offset + i) % 32); 
  }
}

uint32_t HitPattern::getHitPattern(int position) const {
  int offset = position * HitSize;
  uint32_t pattern = 0; 
  for (int i=0; i<HitSize; i++) {
    int pos = offset + i;
    uint32_t word = hitPattern_[pos / 32];
    uint32_t bit = (word >> (pos%32)) & 0x1;
    pattern += bit << i;
  }
  return pattern;
}

bool HitPattern::trackerHitFilter(uint32_t pattern) const {
  if (pattern == 0) return false;
  if (((pattern>>SubDetectorOffset) & SubDetectorMask) == 1) return true;
  return false;
}

bool HitPattern::muonHitFilter(uint32_t pattern) const {
  if (pattern == 0) return false;
  if (((pattern>>SubDetectorOffset) & SubDetectorMask) == 0) return true; 
  return false;
}

uint32_t HitPattern::getSubStructure(uint32_t pattern) const {
  if (pattern == 0) return 999999;
  return ((pattern >> SubstrOffset) & SubstrMask);
}

bool HitPattern::pixelHitFilter(uint32_t pattern) const { 
  if (!trackerHitFilter(pattern)) return false;
  uint32_t substructure = getSubStructure(pattern);
  if (substructure == PixelSubdetector::PixelBarrel || 
      substructure == PixelSubdetector::PixelEndcap) return true; 
  return false;
}

bool HitPattern::pixelBarrelHitFilter(uint32_t pattern) const { 
  if (!trackerHitFilter(pattern)) return false;
  uint32_t substructure = getSubStructure(pattern);
  if (substructure == PixelSubdetector::PixelBarrel) return true; 
  return false;
}

bool HitPattern::pixelEndcapHitFilter(uint32_t pattern) const { 
  if (!trackerHitFilter(pattern)) return false;
  uint32_t substructure = getSubStructure(pattern);
  if (substructure == PixelSubdetector::PixelEndcap) return true; 
  return false;
}

bool HitPattern::stripHitFilter(uint32_t pattern) const { 
  if (!trackerHitFilter(pattern)) return false;
  uint32_t substructure = getSubStructure(pattern);
  if (substructure == StripSubdetector::TIB ||
      substructure == StripSubdetector::TID ||
      substructure == StripSubdetector::TOB ||
      substructure == StripSubdetector::TEC) return true; 
  return false;
}

bool HitPattern::stripTIBHitFilter(uint32_t pattern) const { 
  if (!trackerHitFilter(pattern)) return false;
  uint32_t substructure = getSubStructure(pattern);
  if (substructure == StripSubdetector::TIB) return true; 
  return false;
}

bool HitPattern::stripTIDHitFilter(uint32_t pattern) const { 
  if (!trackerHitFilter(pattern)) return false;
  uint32_t substructure = getSubStructure(pattern);
  if (substructure == StripSubdetector::TID) return true; 
  return false;
}

bool HitPattern::stripTOBHitFilter(uint32_t pattern) const { 
  if (!trackerHitFilter(pattern)) return false;
  uint32_t substructure = getSubStructure(pattern);
  if (substructure == StripSubdetector::TOB) return true; 
  return false;
}

bool HitPattern::stripTECHitFilter(uint32_t pattern) const { 
  if (!trackerHitFilter(pattern)) return false;
  uint32_t substructure = getSubStructure(pattern);
  if (substructure == StripSubdetector::TEC) return true; 
  return false;
}

bool HitPattern::muonDTHitFilter(uint32_t pattern) const { 
  if (!muonHitFilter(pattern)) return false;
  uint32_t substructure = getSubStructure(pattern);
  if (substructure == (uint32_t) MuonSubdetId::DT) return true; 
  return false;
}

bool HitPattern::muonCSCHitFilter(uint32_t pattern) const { 
  if (!muonHitFilter(pattern)) return false;
  uint32_t substructure = getSubStructure(pattern);
  if (substructure == (uint32_t) MuonSubdetId::CSC) return true; 
  return false;
}

bool HitPattern::muonRPCHitFilter(uint32_t pattern) const { 
  if (!muonHitFilter(pattern)) return false;
  uint32_t substructure = getSubStructure(pattern);
  if (substructure == (uint32_t) MuonSubdetId::RPC) return true; 
  return false;
}

uint32_t HitPattern::getLayer(uint32_t pattern) const {
  if (pattern == 0) return 999999;
  return ((pattern>>LayerOffset) & LayerMask);
}

uint32_t HitPattern::getHitType( uint32_t pattern ) const {
  if (pattern == 0) return 999999;
  return ((pattern>>HitTypeOffset) & HitTypeMask);
}

bool HitPattern::validHitFilter(uint32_t pattern) const {
  if (getHitType(pattern) == 0) return true; 
  return false;
}

bool HitPattern::type_1_HitFilter(uint32_t pattern) const {
  if (getHitType(pattern) == 1) return true; 
  return false;
}

bool HitPattern::type_2_HitFilter(uint32_t pattern) const {
  if (getHitType(pattern) == 2) return true; 
  return false;
}

bool HitPattern::type_3_HitFilter(uint32_t pattern) const {
  if (getHitType(pattern) == 3) return true; 
  return false;
}

bool HitPattern::hasValidHitInFirstPixelBarrel() const {
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pixelBarrelHitFilter(pattern)) {
      if (getLayer(pattern) == 1) {
        if (validHitFilter(pattern)) {
          return true;
        }
      }
    }
  }
  return false;
}

int HitPattern::numberOfHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) count++;
  }
  return count;
}

int HitPattern::numberOfValidHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (validHitFilter(pattern)) count++;
    }
  }
  return count;
}

int HitPattern::numberOfValidTrackerHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (validHitFilter(pattern)) {
        if (trackerHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}

int HitPattern::numberOfValidMuonHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (validHitFilter(pattern)) {
        if (muonHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}

int HitPattern::numberOfValidPixelHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (validHitFilter(pattern)) {
        if (pixelHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}

int HitPattern::numberOfValidPixelBarrelHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (validHitFilter(pattern)) {
        if (pixelBarrelHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}

int HitPattern::numberOfValidPixelEndcapHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (validHitFilter(pattern)) {
        if (pixelEndcapHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}

int HitPattern::numberOfValidStripHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (validHitFilter(pattern)) {
        if (stripHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}

int HitPattern::numberOfValidStripTIBHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (validHitFilter(pattern)) {
        if (stripTIBHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}

int HitPattern::numberOfValidStripTIDHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (validHitFilter(pattern)) {
        if (stripTIDHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}

int HitPattern::numberOfValidStripTOBHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (validHitFilter(pattern)) {
        if (stripTOBHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}

int HitPattern::numberOfValidStripTECHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (validHitFilter(pattern)) {
        if (stripTECHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}

int HitPattern::numberOfValidMuonDTHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (validHitFilter(pattern)) {
        if (muonDTHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}

int HitPattern::numberOfValidMuonCSCHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (validHitFilter(pattern)) {
        if (muonCSCHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}

int HitPattern::numberOfValidMuonRPCHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (validHitFilter(pattern)) {
        if (muonRPCHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}

int HitPattern::numberOfLostHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (!validHitFilter(pattern)) count++;
    }
  }
  return count;
}

int HitPattern::numberOfLostTrackerHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (!validHitFilter(pattern)) {
        if (trackerHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}

int HitPattern::numberOfLostMuonHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (!validHitFilter(pattern)) {
        if (muonHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}

int HitPattern::numberOfLostPixelHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (!validHitFilter(pattern)) {
        if (pixelHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}

int HitPattern::numberOfLostPixelBarrelHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (!validHitFilter(pattern)) {
        if (pixelBarrelHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}

int HitPattern::numberOfLostPixelEndcapHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (!validHitFilter(pattern)) {
        if (pixelEndcapHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}

int HitPattern::numberOfLostStripHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (!validHitFilter(pattern)) {
        if (stripHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}

int HitPattern::numberOfLostStripTIBHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (!validHitFilter(pattern)) {
        if (stripTIBHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}

int HitPattern::numberOfLostStripTIDHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (!validHitFilter(pattern)) {
        if (stripTIDHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}

int HitPattern::numberOfLostStripTOBHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (!validHitFilter(pattern)) {
        if (stripTOBHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}

int HitPattern::numberOfLostStripTECHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (!validHitFilter(pattern)) {
        if (stripTECHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}

int HitPattern::numberOfLostMuonDTHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (!validHitFilter(pattern)) {
        if (muonDTHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}

int HitPattern::numberOfLostMuonCSCHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (!validHitFilter(pattern)) {
        if (muonCSCHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}

int HitPattern::numberOfLostMuonRPCHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (!validHitFilter(pattern)) {
        if (muonRPCHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}
