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


uint32_t HitPattern::encode(const TrackingRecHit & hit, unsigned int i){
  
  // ignore the rec hit if the number of hit is larger than the max
  if (i >= 32 * PatternSize / HitSize) return 0;

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
      layer = ((DTLayerId(id.rawId()).station()-1)<<2) + DTLayerId(id.rawId()).superLayer();
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

  return pattern;
}


void HitPattern::setHitPattern(int position, uint32_t pattern) {
  int offset = position * HitSize;
  for (int i=0; i<HitSize; i++) {
    int pos = offset + i;
    uint32_t bit = (pattern >> i) & 0x1;
    hitPattern_[pos / 32] += bit << ((offset + i) % 32); 
  }
}

void HitPattern::appendHit(const TrackingRecHit & hit){

  // get rec hit det id and rec hit type
  DetId id = hit.geographicalId();
  uint32_t detid = id.det();
  uint32_t subdet = id.subdetId();

  std::vector<const TrackingRecHit*> hits;


  if (detid == DetId::Tracker)
    hits.push_back(&hit);
   
  if (detid == DetId::Muon) {

    if (subdet == (uint32_t) MuonSubdetId::DT){

      // DT rechit (granularity 2)
      if(hit.dimension() == 1)
	hits.push_back(&hit);
      
      // 4D segment (granularity 0) 
      else if(hit.dimension() > 1){ // Both 2 and 4 dim cases. MB4s have 2D, but formatted in 4D segments 
	std::vector<const TrackingRecHit*> seg2D = hit.recHits(); // 4D --> 2D
	// load 1D hits (2D --> 1D)
	for(std::vector<const TrackingRecHit*>::const_iterator it = seg2D.begin(); it != seg2D.end(); ++it){
	  std::vector<const TrackingRecHit*> hits1D =  (*it)->recHits();
	  copy(hits1D.begin(),hits1D.end(),back_inserter(hits));
	}
      }
    }

    else if (subdet == (uint32_t) MuonSubdetId::CSC){
      
      // CSC rechit (granularity 2)
      if(hit.dimension() == 2)
	hits.push_back(&hit);

      // 4D segment (granularity 0) 
      if(hit.dimension() == 4)
	hits = hit.recHits(); // load 2D hits (4D --> 1D)
    }
   
    else if (subdet == (uint32_t) MuonSubdetId::RPC) {
      hits.push_back(&hit);
    }
  }

  unsigned int i =  numberOfHits();
  for(std::vector<const TrackingRecHit*>::const_iterator it = hits.begin(); it != hits.end(); ++it)
    set(**it,i++);


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





bool HitPattern::hasValidHitInFirstPixelBarrel() const {
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern == 0) break;
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

bool HitPattern::hasValidHitInFirstPixelEndcap() const {
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern == 0) break;
    if (pixelEndcapHitFilter(pattern)) {
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
    if (pattern == 0) break;
    ++count;
  }
  return count;
}


int HitPattern::numberOfValidStripLayersWithMonoAndStereo () const {
  static const int nHits = (PatternSize * 32) / HitSize;
  bool hasMono[SubstrMask + 1][LayerMask + 1];
  //     printf("sizeof(hasMono) = %d\n", sizeof(hasMono));
  memset(hasMono, 0, sizeof(hasMono));
  bool hasStereo[SubstrMask + 1][LayerMask + 1];
  memset(hasStereo, 0, sizeof(hasStereo));
  // mark which layers have mono/stereo hits
  for (int i = 0; i < nHits; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern == 0) break;
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
  // count how many layers have mono and stereo hits
  int count = 0;
  for (int i = 0; i < SubstrMask + 1; ++i) 
    for (int j = 0; j < LayerMask + 1; ++j)
      if (hasMono[i][j] && hasStereo[i][j])
	count++;
  return count;
}

uint32_t HitPattern::getTrackerLayerCase(uint32_t substr, uint32_t layer) const {
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
    if ((pattern & mask) == tk_substr_layer) {
      uint32_t hitType = (pattern >> HitTypeOffset) & HitTypeMask; // 0,1,2,3
      if (hitType < layerCase) {
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



int HitPattern::pixelBarrelLayersWithMeasurement() const {
  int count = 0;
  uint32_t substr = PixelSubdetector::PixelBarrel;
  uint32_t NPixBarrel = 4;
  for (uint32_t layer=1; layer<=NPixBarrel; layer++) {
    if (getTrackerLayerCase(substr, layer) == 0) count++;
  }
  return count;
}

int HitPattern::pixelEndcapLayersWithMeasurement() const {
  int count = 0;
  uint32_t substr = PixelSubdetector::PixelEndcap;
  uint32_t NPixForward = 3;
  for (uint32_t layer=1; layer<=NPixForward; layer++) {
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


int HitPattern::pixelBarrelLayersWithoutMeasurement() const {
  int count = 0;
  uint32_t substr = PixelSubdetector::PixelBarrel;
  uint32_t NPixBarrel = 4;
  for (uint32_t layer=1; layer<=NPixBarrel; layer++) {
    if (getTrackerLayerCase(substr, layer) == 1) count++;
  }
  return count;
}

int HitPattern::pixelEndcapLayersWithoutMeasurement() const {
  int count = 0;
  uint32_t substr = PixelSubdetector::PixelEndcap;
  uint32_t NPixForward = 3;
  for (uint32_t layer=1; layer<=NPixForward; layer++) {
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


int HitPattern::pixelBarrelLayersTotallyOffOrBad() const {
  int count = 0;
  uint32_t substr = PixelSubdetector::PixelBarrel;
  uint32_t NPixBarrel = 4;
  for (uint32_t layer=1; layer<=NPixBarrel; layer++) {
    if (getTrackerLayerCase(substr, layer) == 2) count++;
  }
  return count;
}

int HitPattern::pixelEndcapLayersTotallyOffOrBad() const {
  int count = 0;
  uint32_t substr = PixelSubdetector::PixelEndcap;
  uint32_t NPixForward = 3;
  for (uint32_t layer=1; layer<=NPixForward; layer++) {
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


int HitPattern::pixelBarrelLayersNull() const {
  int count = 0;
  uint32_t substr = PixelSubdetector::PixelBarrel;
  uint32_t NPixBarrel = 4;
  for (uint32_t layer=1; layer<=NPixBarrel; layer++) {
    if (getTrackerLayerCase(substr, layer) == 999999) count++;
  }
  return count;
}

int HitPattern::pixelEndcapLayersNull() const {
  int count = 0;
  uint32_t substr = PixelSubdetector::PixelEndcap;
  uint32_t NPixForward = 3;
  for (uint32_t layer=1; layer<=NPixForward; layer++) {
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

int HitPattern::numberOfDTStationsWithBothViews() const {
  int stations[4][2] = { {0,0}, {0,0}, {0,0}, {0,0} };
  for (int i=0; i<(PatternSize * 32) / HitSize; i++) {
    uint32_t pattern = getHitPattern(i);
    if (pattern == 0) break;
    if (muonDTHitFilter(pattern) && validHitFilter(pattern)) {
        stations[getMuonStation(pattern)-1][getDTSuperLayer(pattern) == 2] = 1;
    }
  }
  return stations[0][0]*stations[0][1] + stations[1][0]*stations[1][1] + stations[2][0]*stations[2][1] + stations[3][0]*stations[3][1];
}



