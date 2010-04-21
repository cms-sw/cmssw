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
      layer = ((DTLayerId(id.rawId()).station()-1)<<2) + (DTLayerId(id.rawId()).superLayer()-1);
    else if (subdet == (uint32_t) MuonSubdetId::CSC)
      layer = ((CSCDetId(id.rawId()).station()-1)<<2) +  (CSCDetId(id.rawId()).ring()-1);
    else if (subdet == (uint32_t) MuonSubdetId::RPC) {
      RPCDetId rpcid(id.rawId());
      layer = ((rpcid.station()-1)<<2) + abs(rpcid.region());
      if (rpcid.station() <= 2) layer += 2*(rpcid.layer()-1);
    }
  }
  pattern += (layer&LayerMask)<<LayerOffset;

  // adding mono/stereo bit
  uint32_t side = 0;
  if (detid == DetId::Tracker) {
       side = isStereo(id);
  } else if (detid == DetId::Muon) {
       side = 0;
  }
  pattern += (side&SideMask)<<SideOffset;

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
/* Note: you are not taking a consecutive sequence of HitSize bits starting from position*HitSize
         as the bit order in the words are reversed. 
         e.g. if position = 0 you take the lowest 10 bits of the first word.
    
         I hope this can clarify what is the memory layout of such thing

 straight 01234567890123456789012345678901 | 23456789012345678901234567890123 | 4567
 (global) 0         1         2         3  | 3       4         5         6    | 6  
 words    [--------------0---------------] | [--------------1---------------] | [---   
 word     01234567890123456789012345678901 | 01234567890123456789012345678901 | 0123
  (str)   0         1         2         3  | 0         1         2         3  | 0
          [--------------0---------------] | [--------------1---------------] | [---   
 word     10987654321098765432109876543210 | 10987654321098765432109876543210 | 1098
  (rev)   32         21        10        0 | 32         21        10        0 | 32  
 reverse  10987654321098765432109876543210 | 32109876543210987654321098765432 | 5432
          32         21        10        0 | 6  65        54        43      3   9

         ugly enough, but it's not my fault, I was not even in CMS at that time   [gpetrucc] */
  uint16_t bitEndOffset = (position+1) * HitSize;
  uint8_t secondWord   = (bitEndOffset >> 5);
  uint8_t secondWordBits = bitEndOffset & (32-1); // that is, bitEndOffset % 32
  if (secondWordBits >= HitSize) { // full block is in this word
      uint8_t lowBitsToTrash = secondWordBits - HitSize;
      uint32_t myResult = (hitPattern_[secondWord] >> lowBitsToTrash) & ((1 << HitSize)-1);
      return myResult;
  } else {
      uint8_t  firstWordBits   = HitSize - secondWordBits;
      uint32_t firstWordBlock  = hitPattern_[secondWord-1] >> (32-firstWordBits);
      uint32_t secondWordBlock = hitPattern_[secondWord] & ((1<<secondWordBits)-1);
      uint32_t myResult = firstWordBlock + (secondWordBlock << firstWordBits);
      return myResult;
  }
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

uint32_t HitPattern::getSubSubStructure(uint32_t pattern) const {
  if (pattern == 0) return 999999;
  return ((pattern>>LayerOffset) & LayerMask);
}


uint32_t HitPattern::getSide (uint32_t pattern) 
{
     if (pattern == 0) return 999999;
     return (pattern >> SideOffset) & SideMask;
}

uint32_t HitPattern::getHitType( uint32_t pattern ) const {
  if (pattern == 0) return 999999;
  return ((pattern>>HitTypeOffset) & HitTypeMask);
}

uint32_t HitPattern::getMuonStation(uint32_t pattern) const {
  return (getSubSubStructure(pattern)>>2) + 1;
}

uint32_t HitPattern::getDTSuperLayer(uint32_t pattern) const {
  return (getSubSubStructure(pattern) & 3) + 1;
}

uint32_t HitPattern::getCSCRing(uint32_t pattern) const {
  return (getSubSubStructure(pattern) & 3) + 1;
}

uint32_t HitPattern::getRPCLayer(uint32_t pattern) const {
    uint32_t sss = getSubSubStructure(pattern), stat = sss >> 2;
    if (stat <= 1) return ((sss >> 1) & 1) + 1;
    else return 0;
}

uint32_t HitPattern::getRPCregion(uint32_t pattern) const {
    return getSubSubStructure(pattern) & 1;
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
    if (pattern == 0) break;
    //if (pattern != 0) {
      if (validHitFilter(pattern)) count++;
    //}
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
      if (type_1_HitFilter(pattern)) count++;
    }
  }
  return count;
}

int HitPattern::numberOfLostTrackerHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (type_1_HitFilter(pattern)) {
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
      if (type_1_HitFilter(pattern)) {
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
      if (type_1_HitFilter(pattern)) {
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
      if (type_1_HitFilter(pattern)) {
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
      if (type_1_HitFilter(pattern)) {
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
      if (type_1_HitFilter(pattern)) {
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
      if (type_1_HitFilter(pattern)) {
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
      if (type_1_HitFilter(pattern)) {
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
      if (type_1_HitFilter(pattern)) {
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
      if (type_1_HitFilter(pattern)) {
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
      if (type_1_HitFilter(pattern)) {
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
      if (type_1_HitFilter(pattern)) {
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
      if (type_1_HitFilter(pattern)) {
        if (muonRPCHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}

int HitPattern::numberOfBadHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (type_3_HitFilter(pattern)) count++;
    }
  }
  return count;
}

int HitPattern::numberOfBadMuonHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (type_3_HitFilter(pattern)) {
        if (muonHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}

int HitPattern::numberOfBadMuonDTHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (type_3_HitFilter(pattern)) {
        if (muonDTHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}

int HitPattern::numberOfBadMuonCSCHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (type_3_HitFilter(pattern)) {
        if (muonCSCHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}

int HitPattern::numberOfBadMuonRPCHits() const {
  int count = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern != 0) {
      if (type_3_HitFilter(pattern)) {
        if (muonRPCHitFilter(pattern)) count++;
      }
    }
  }
  return count;
}

int HitPattern::numberOfValidStripLayersWithMonoAndStereo () const 
{
     static const int nHits = (PatternSize * 32) / HitSize;
     bool hasMono[SubstrMask + 1][LayerMask + 1];
     //     printf("sizeof(hasMono) = %d\n", sizeof(hasMono));
     memset(hasMono, 0, sizeof(hasMono));
     bool hasStereo[SubstrMask + 1][LayerMask + 1];
     memset(hasStereo, 0, sizeof(hasStereo));
     // mark which layers have mono/stereo hits
     for (int i = 0; i < nHits; i++) {
	  uint32_t pattern = getHitPattern(i);
	  if (pattern != 0) {
	       if (validHitFilter(pattern) && stripHitFilter(pattern)) {
		    switch (getSide(pattern)) {
		    case 0: // mono
			 hasMono[getSubStructure(pattern)][getLayer(pattern)] 
			      = true;
			 break;
		    case 1: // stereo
			 hasStereo[getSubStructure(pattern)][getLayer(pattern)]
			      = true;
			 break;
		    default:
			 break;
		    }
	       }
	  }
     }
     // count how many layers have mono and stereo hits
     int count = 0;
     for (int i = 0; i < SubstrMask + 1; ++i) 
	  for (int j = 0; j < LayerMask + 1; ++j)
	       if (hasMono[i][j] && hasStereo[i][j])
		    count++;
     return count;
}

uint32_t HitPattern::getTrackerLayerCase(uint32_t substr, uint32_t layer) const
{
  uint32_t tk_substr_layer = 
    (1 << SubDetectorOffset) +
    ((substr & SubstrMask) << SubstrOffset) +
    ((layer & LayerMask) << LayerOffset);

  uint32_t mask =
    (SubDetectorMask << SubDetectorOffset) +
    (SubstrMask << SubstrOffset) +
    (LayerMask << LayerOffset);

  // crossed
  //   layer case 0: valid + (missing, off, bad) ==> with measurement
  //   layer case 1: missing + (off, bad) ==> without measurement
  //   layer case 2: off, bad ==> totally off or bad, cannot say much
  // not crossed
  //   layer case 999999: track outside acceptance or in gap ==> null

  uint32_t layerCase = 999999;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++)
  {
    uint32_t pattern = getHitPattern(i);
    if (pattern == 0) break;
    if ((pattern & mask) == tk_substr_layer)
    {
      uint32_t hitType = (pattern >> HitTypeOffset) & HitTypeMask; // 0,1,2,3
      if (hitType < layerCase)
      {
        layerCase = hitType;
        if (hitType == 3) layerCase = 2;
      }
    }
  }
  return layerCase;
}

uint32_t HitPattern::getTrackerMonoStereo (uint32_t substr, uint32_t layer) const
{
  uint32_t tk_substr_layer = 
    (1 << SubDetectorOffset) +
    ((substr & SubstrMask) << SubstrOffset) +
    ((layer & LayerMask) << LayerOffset);

  uint32_t mask =
    (SubDetectorMask << SubDetectorOffset) +
    (SubstrMask << SubstrOffset) +
    (LayerMask << LayerOffset);

  //	0:		neither a valid mono nor a valid stereo hit
  //    MONO:		valid mono hit
  //	STEREO:		valid stereo hit
  //	MONO | STEREO:	both
  uint32_t monoStereo = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++)
  {
    uint32_t pattern = getHitPattern(i);
    if (pattern == 0) break;
    if ((pattern & mask) == tk_substr_layer)
    {
      uint32_t hitType = (pattern >> HitTypeOffset) & HitTypeMask; // 0,1,2,3
      if (hitType == 0) { // valid hit
	   switch (getSide(pattern)) {
	   case 0: // mono
		monoStereo |= MONO;
		break;
	   case 1: // stereo
		monoStereo |= STEREO;
		break;
	   default:
		break;
	   }
      }
    }
  }
  return monoStereo;
}

int HitPattern::trackerLayersWithMeasurement() const {
  return pixelLayersWithMeasurement() + 
         stripLayersWithMeasurement();
}

int HitPattern::pixelLayersWithMeasurement() const {
  return pixelBarrelLayersWithMeasurement() +
         pixelEndcapLayersWithMeasurement();
}

int HitPattern::stripLayersWithMeasurement() const {
  return stripTIBLayersWithMeasurement() + 
         stripTIDLayersWithMeasurement() +
         stripTOBLayersWithMeasurement() + 
         stripTECLayersWithMeasurement();
}

int HitPattern::pixelBarrelLayersWithMeasurement() const {
  int count = 0;
  uint32_t substr = PixelSubdetector::PixelBarrel;
  for (uint32_t layer=1; layer<=3; layer++) {
    if (getTrackerLayerCase(substr, layer) == 0) count++;
  }
  return count;
}

int HitPattern::pixelEndcapLayersWithMeasurement() const {
  int count = 0;
  uint32_t substr = PixelSubdetector::PixelEndcap;
  for (uint32_t layer=1; layer<=2; layer++) {
    if (getTrackerLayerCase(substr, layer) == 0) count++;
  }
  return count;
}

int HitPattern::stripTIBLayersWithMeasurement() const {
  int count = 0;
  uint32_t substr = StripSubdetector::TIB;
  for (uint32_t layer=1; layer<=4; layer++) {
    if (getTrackerLayerCase(substr, layer) == 0) count++;
  }
  return count;
}

int HitPattern::stripTIDLayersWithMeasurement() const {
  int count = 0;
  uint32_t substr = StripSubdetector::TID;
  for (uint32_t layer=1; layer<=3; layer++) {
    if (getTrackerLayerCase(substr, layer) == 0) count++;
  }
  return count;
}

int HitPattern::stripTOBLayersWithMeasurement() const {
  int count = 0;
  uint32_t substr = StripSubdetector::TOB;
  for (uint32_t layer=1; layer<=6; layer++) {
    if (getTrackerLayerCase(substr, layer) == 0) count++;
  }
  return count;
}

int HitPattern::stripTECLayersWithMeasurement() const {
  int count = 0;
  uint32_t substr = StripSubdetector::TEC;
  for (uint32_t layer=1; layer<=9; layer++) {
    if (getTrackerLayerCase(substr, layer) == 0) count++;
  }
  return count;
}

int HitPattern::trackerLayersWithoutMeasurement() const {
  return pixelLayersWithoutMeasurement() + 
         stripLayersWithoutMeasurement();
}

int HitPattern::pixelLayersWithoutMeasurement() const {
  return pixelBarrelLayersWithoutMeasurement() +
         pixelEndcapLayersWithoutMeasurement();
}

int HitPattern::stripLayersWithoutMeasurement() const {
  return stripTIBLayersWithoutMeasurement() + 
         stripTIDLayersWithoutMeasurement() +
         stripTOBLayersWithoutMeasurement() + 
         stripTECLayersWithoutMeasurement();
}

int HitPattern::pixelBarrelLayersWithoutMeasurement() const {
  int count = 0;
  uint32_t substr = PixelSubdetector::PixelBarrel;
  for (uint32_t layer=1; layer<=3; layer++) {
    if (getTrackerLayerCase(substr, layer) == 1) count++;
  }
  return count;
}

int HitPattern::pixelEndcapLayersWithoutMeasurement() const {
  int count = 0;
  uint32_t substr = PixelSubdetector::PixelEndcap;
  for (uint32_t layer=1; layer<=2; layer++) {
    if (getTrackerLayerCase(substr, layer) == 1) count++;
  }
  return count;
}

int HitPattern::stripTIBLayersWithoutMeasurement() const {
  int count = 0;
  uint32_t substr = StripSubdetector::TIB;
  for (uint32_t layer=1; layer<=4; layer++) {
    if (getTrackerLayerCase(substr, layer) == 1) count++;
  }
  return count;
}

int HitPattern::stripTIDLayersWithoutMeasurement() const {
  int count = 0;
  uint32_t substr = StripSubdetector::TID;
  for (uint32_t layer=1; layer<=3; layer++) {
    if (getTrackerLayerCase(substr, layer) == 1) count++;
  }
  return count;
}

int HitPattern::stripTOBLayersWithoutMeasurement() const {
  int count = 0;
  uint32_t substr = StripSubdetector::TOB;
  for (uint32_t layer=1; layer<=6; layer++) {
    if (getTrackerLayerCase(substr, layer) == 1) count++;
  }
  return count;
}

int HitPattern::stripTECLayersWithoutMeasurement() const {
  int count = 0;
  uint32_t substr = StripSubdetector::TEC;
  for (uint32_t layer=1; layer<=9; layer++) {
    if (getTrackerLayerCase(substr, layer) == 1) count++;
  }
  return count;
}

int HitPattern::trackerLayersTotallyOffOrBad() const {
  return pixelLayersTotallyOffOrBad() + 
         stripLayersTotallyOffOrBad();
}

int HitPattern::pixelLayersTotallyOffOrBad() const {
  return pixelBarrelLayersTotallyOffOrBad() +
         pixelEndcapLayersTotallyOffOrBad();
}

int HitPattern::stripLayersTotallyOffOrBad() const {
  return stripTIBLayersTotallyOffOrBad() + 
         stripTIDLayersTotallyOffOrBad() +
         stripTOBLayersTotallyOffOrBad() + 
         stripTECLayersTotallyOffOrBad();
}

int HitPattern::pixelBarrelLayersTotallyOffOrBad() const {
  int count = 0;
  uint32_t substr = PixelSubdetector::PixelBarrel;
  for (uint32_t layer=1; layer<=3; layer++) {
    if (getTrackerLayerCase(substr, layer) == 2) count++;
  }
  return count;
}

int HitPattern::pixelEndcapLayersTotallyOffOrBad() const {
  int count = 0;
  uint32_t substr = PixelSubdetector::PixelEndcap;
  for (uint32_t layer=1; layer<=2; layer++) {
    if (getTrackerLayerCase(substr, layer) == 2) count++;
  }
  return count;
}

int HitPattern::stripTIBLayersTotallyOffOrBad() const {
  int count = 0;
  uint32_t substr = StripSubdetector::TIB;
  for (uint32_t layer=1; layer<=4; layer++) {
    if (getTrackerLayerCase(substr, layer) == 2) count++;
  }
  return count;
}

int HitPattern::stripTIDLayersTotallyOffOrBad() const {
  int count = 0;
  uint32_t substr = StripSubdetector::TID;
  for (uint32_t layer=1; layer<=3; layer++) {
    if (getTrackerLayerCase(substr, layer) == 2) count++;
  }
  return count;
}

int HitPattern::stripTOBLayersTotallyOffOrBad() const {
  int count = 0;
  uint32_t substr = StripSubdetector::TOB;
  for (uint32_t layer=1; layer<=6; layer++) {
    if (getTrackerLayerCase(substr, layer) == 2) count++;
  }
  return count;
}

int HitPattern::stripTECLayersTotallyOffOrBad() const {
  int count = 0;
  uint32_t substr = StripSubdetector::TEC;
  for (uint32_t layer=1; layer<=9; layer++) {
    if (getTrackerLayerCase(substr, layer) == 2) count++;
  }
  return count;
}

int HitPattern::trackerLayersNull() const {
  return pixelLayersNull() + 
         stripLayersNull();
}

int HitPattern::pixelLayersNull() const {
  return pixelBarrelLayersNull() +
         pixelEndcapLayersNull();
}

int HitPattern::stripLayersNull() const {
  return stripTIBLayersNull() + 
         stripTIDLayersNull() +
         stripTOBLayersNull() + 
         stripTECLayersNull();
}

int HitPattern::pixelBarrelLayersNull() const {
  int count = 0;
  uint32_t substr = PixelSubdetector::PixelBarrel;
  for (uint32_t layer=1; layer<=3; layer++) {
    if (getTrackerLayerCase(substr, layer) == 999999) count++;
  }
  return count;
}

int HitPattern::pixelEndcapLayersNull() const {
  int count = 0;
  uint32_t substr = PixelSubdetector::PixelEndcap;
  for (uint32_t layer=1; layer<=2; layer++) {
    if (getTrackerLayerCase(substr, layer) == 999999) count++;
  }
  return count;
}

int HitPattern::stripTIBLayersNull() const {
  int count = 0;
  uint32_t substr = StripSubdetector::TIB;
  for (uint32_t layer=1; layer<=4; layer++) {
    if (getTrackerLayerCase(substr, layer) == 999999) count++;
  }
  return count;
}

int HitPattern::stripTIDLayersNull() const {
  int count = 0;
  uint32_t substr = StripSubdetector::TID;
  for (uint32_t layer=1; layer<=3; layer++) {
    if (getTrackerLayerCase(substr, layer) == 999999) count++;
  }
  return count;
}

int HitPattern::stripTOBLayersNull() const {
  int count = 0;
  uint32_t substr = StripSubdetector::TOB;
  for (uint32_t layer=1; layer<=6; layer++) {
    if (getTrackerLayerCase(substr, layer) == 999999) count++;
  }
  return count;
}

int HitPattern::stripTECLayersNull() const {
  int count = 0;
  uint32_t substr = StripSubdetector::TEC;
  for (uint32_t layer=1; layer<=9; layer++) {
    if (getTrackerLayerCase(substr, layer) == 999999) count++;
  }
  return count;
}

void HitPattern::printHitPattern (int position, std::ostream &stream) const
{
     uint32_t pattern = getHitPattern(position);
     stream << "\t";
     if (muonHitFilter(pattern))
	  stream << "muon";
     if (trackerHitFilter(pattern))
	  stream << "tracker";
     stream << "\tsubstructure " << getSubStructure(pattern);
     if (muonHitFilter(pattern)) {
         stream << "\tstation " << getMuonStation(pattern);
         if (muonDTHitFilter(pattern)) { 
            stream << "\tdt superlayer " << getDTSuperLayer(pattern); 
         } else if (muonCSCHitFilter(pattern)) { 
            stream << "\tcsc ring " << getCSCRing(pattern); 
         } else if (muonRPCHitFilter(pattern)) {
            stream << "\trpc " << (getRPCregion(pattern) ? "endcaps" : "barrel") << ", layer " << getRPCLayer(pattern); 
         } else {
            stream << "(UNKNOWN Muon SubStructure!) \tsubsubstructure " << getSubStructure(pattern);
         }
     } else {
         stream << "\tlayer " << getLayer(pattern);
     }
     stream << "\thit type " << getHitType(pattern);
     stream << std::endl;
}

void HitPattern::print (std::ostream &stream) const
{
     stream << "HitPattern" << std::endl;
     for (int i = 0; i < numberOfHits(); i++) 
	  printHitPattern(i, stream);
     std::ios_base::fmtflags flags = stream.flags();
     stream.setf ( std::ios_base::hex, std::ios_base::basefield );  
     stream.setf ( std::ios_base::showbase );               
     for (int i = 0; i < numberOfHits(); i++) {
	  uint32_t pattern = getHitPattern(i);
	  stream << pattern << std::endl;
     }
     stream.flags(flags);
}

uint32_t HitPattern::isStereo (DetId i) 
{
     switch (i.det()) {
     case DetId::Tracker:
	  switch (i.subdetId()) {
	  case PixelSubdetector::PixelBarrel:
	  case PixelSubdetector::PixelEndcap:
	       return 0;
	  case StripSubdetector::TIB:
	  {
	       TIBDetId id = i;
	       return id.isStereo();
	  }
	  case StripSubdetector::TID:
	  {
	       TIDDetId id = i;
	       return id.isStereo();
	  }
	  case StripSubdetector::TOB:
	  {
	       TOBDetId id = i;
	       return id.isStereo();
	  }
	  case StripSubdetector::TEC:
	  {
	       TECDetId id = i;
	       return id.isStereo();
	  }
	  default:
	       return 0;
	  }
	  break;
     default:
	  return 0;
     }
}

int  HitPattern::muonStations(int subdet, int hitType) const {
  int stations[4] = { 0,0,0,0 };
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern == 0) break;
    if (muonHitFilter(pattern) &&
        (subdet  == 0  || int(getSubStructure(pattern)) == subdet) &&
        (hitType == -1 || int(getHitType(pattern))      == hitType)) {
        stations[getMuonStation(pattern)-1] = 1;
    }
  }
  return stations[0]+stations[1]+stations[2]+stations[3];
}

int HitPattern::muonStationsWithValidHits() const { return muonStations(0, 0); }
int HitPattern::muonStationsWithBadHits()   const { return muonStations(0, 3); }
int HitPattern::muonStationsWithAnyHits()   const { return muonStations(0,-1); }
int HitPattern::dtStationsWithValidHits()   const { return muonStations(1, 0); }
int HitPattern::dtStationsWithBadHits()     const { return muonStations(1, 3); }
int HitPattern::dtStationsWithAnyHits()     const { return muonStations(1,-1); }
int HitPattern::cscStationsWithValidHits()  const { return muonStations(2, 0); }
int HitPattern::cscStationsWithBadHits()    const { return muonStations(2, 3); }
int HitPattern::cscStationsWithAnyHits()    const { return muonStations(2,-1); }
int HitPattern::rpcStationsWithValidHits()  const { return muonStations(3, 0); }
int HitPattern::rpcStationsWithBadHits()    const { return muonStations(3, 3); }
int HitPattern::rpcStationsWithAnyHits()    const { return muonStations(3,-1); }

int HitPattern::innermostMuonStationWithHits(int hitType) const {
  int ret = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern == 0) break;
    if (muonHitFilter(pattern) &&
        (hitType == -1 || int(getHitType(pattern)) == hitType)) {
        int stat = getMuonStation(pattern);
        if (ret == 0 || stat < ret) ret = stat;
    }
  }
  return ret;
}

int HitPattern::outermostMuonStationWithHits(int hitType) const {
  int ret = 0;
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern == 0) break;
    if (muonHitFilter(pattern) &&
        (hitType == -1 || int(getHitType(pattern)) == hitType)) {
        int stat = getMuonStation(pattern);
        if (ret == 0 || stat > ret) ret = stat;
    }
  }
  return ret;
}

int HitPattern::innermostMuonStationWithValidHits() const { return innermostMuonStationWithHits(0);  }
int HitPattern::innermostMuonStationWithBadHits()   const { return innermostMuonStationWithHits(3);  }
int HitPattern::innermostMuonStationWithAnyHits()   const { return innermostMuonStationWithHits(-1); }
int HitPattern::outermostMuonStationWithValidHits() const { return outermostMuonStationWithHits(0);  }
int HitPattern::outermostMuonStationWithBadHits()   const { return outermostMuonStationWithHits(3);  }
int HitPattern::outermostMuonStationWithAnyHits()   const { return outermostMuonStationWithHits(-1); }

int HitPattern::numberOfDTStationsWithRPhiView() const {
  int stations[4] = { 0,0,0,0 };
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern == 0) break;
    if (muonDTHitFilter(pattern) && validHitFilter(pattern) && getDTSuperLayer(pattern) != 2) {
        stations[getMuonStation(pattern)-1] = 1;
    }
  }
  return stations[0]+stations[1]+stations[2]+stations[3];
}

int HitPattern::numberOfDTStationsWithRZView() const {
  int stations[4] = { 0,0,0,0 };
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern == 0) break;
    if (muonDTHitFilter(pattern) && validHitFilter(pattern) && getDTSuperLayer(pattern) == 2) {
        stations[getMuonStation(pattern)-1] = 1;
    }
  }
  return stations[0]+stations[1]+stations[2]+stations[3];
}


