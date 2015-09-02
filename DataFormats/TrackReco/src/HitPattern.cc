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
#include "DataFormats/MuonDetId/interface/GEMDetId.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include<bitset>

using namespace reco;

HitPattern::HitPattern() :
    hitCount(0),
    beginTrackHits(0),
    endTrackHits(0),
    beginInner(0),
    endInner(0),
    beginOuter(0),
    endOuter(0)
{
    memset(hitPattern, HitPattern::EMPTY_PATTERN, sizeof(uint16_t) * HitPattern::ARRAY_LENGTH);
}

HitPattern::HitPattern(const HitPattern &other) :
    hitCount(other.hitCount),
    beginTrackHits(other.beginTrackHits),
    endTrackHits(other.endTrackHits),
    beginInner(other.beginInner),
    endInner(other.endInner),
    beginOuter(other.beginOuter),
    endOuter(other.endOuter)
{
    memcpy(this->hitPattern, other.hitPattern, sizeof(uint16_t) * HitPattern::ARRAY_LENGTH);
}

HitPattern::~HitPattern()
{
    ;
}

HitPattern & HitPattern::operator=(const HitPattern &other)
{
    if (this == &other) {
        return *this;
    }

    this->hitCount = other.hitCount;

    this->beginTrackHits = other.beginTrackHits;
    this->endTrackHits = other.endTrackHits;

    this->beginInner = other.beginInner;
    this->endInner = other.endInner;

    this->beginOuter = other.beginOuter;
    this->endOuter = other.endOuter;

    memcpy(this->hitPattern, other.hitPattern, sizeof(uint16_t) * HitPattern::ARRAY_LENGTH);

    return *this;
}

void HitPattern::clear(void)
{
    this->hitCount = 0;
    this->beginTrackHits = 0;
    this->endTrackHits = 0;
    this->beginInner = 0;
    this->endInner = 0;
    this->beginOuter = 0;
    this->endOuter = 0;

    memset(this->hitPattern, EMPTY_PATTERN, sizeof(uint16_t) * HitPattern::ARRAY_LENGTH);
}

bool HitPattern::appendHit(const TrackingRecHitRef &ref, const TrackerTopology& ttopo)
{
    return appendHit(*ref, ttopo);
}

uint16_t HitPattern::encode(const TrackingRecHit &hit, const TrackerTopology& ttopo)
{
    return encode(hit.geographicalId(), hit.getType(), ttopo);
}

namespace {
    uint16_t encodeMuonLayer(const DetId& id) {
        uint16_t detid = id.det();
        uint16_t subdet = id.subdetId();

        uint16_t layer = 0x0;
        if (detid == DetId::Muon) {
            switch (subdet) {
            case MuonSubdetId::DT:
                layer = ((DTLayerId(id.rawId()).station() - 1) << 2);
                layer |= DTLayerId(id.rawId()).superLayer();
                break;
            case MuonSubdetId::CSC:
                layer = ((CSCDetId(id.rawId()).station() - 1) << 2);
                layer |= (CSCDetId(id.rawId()).ring() - 1);
                break;
            case MuonSubdetId::RPC: 
                {
                    RPCDetId rpcid(id.rawId());
                    layer = ((rpcid.station() - 1) << 2);
                    layer |= (rpcid.station() <= 2) ? ((rpcid.layer() - 1) << 1) : 0x0;
                    layer |= abs(rpcid.region());
                }
                break;
            case MuonSubdetId::GEM:
            {
              GEMDetId gemid(id.rawId());
              layer = ((gemid.station()-1)<<2);
              layer |= abs(gemid.layer()-1);
            }
            break;
            }
        }
        return layer;
    }
}

uint16_t HitPattern::encode(const DetId &id, TrackingRecHit::Type hitType, const TrackerTopology& ttopo)
{
    uint16_t detid = id.det();
    uint16_t subdet = id.subdetId();

    // adding layer/disk/wheel bits
    uint16_t layer = 0x0;
    if (detid == DetId::Tracker) {
        layer = ttopo.layer(id);
    } else if (detid == DetId::Muon) {
        layer = encodeMuonLayer(id);
    }

    // adding mono/stereo bit
    uint16_t side = 0x0;
    if (detid == DetId::Tracker) {
        side = isStereo(id, ttopo);
    } else if (detid == DetId::Muon) {
        side = 0x0;
    }

    return encode(detid, subdet, layer, side, hitType);
}

uint16_t HitPattern::encode(uint16_t det, uint16_t subdet, uint16_t layer, uint16_t side, TrackingRecHit::Type hitType) {
    uint16_t pattern = HitPattern::EMPTY_PATTERN;

    // adding tracker/muon detector bit
    pattern |= (det & SubDetectorMask) << SubDetectorOffset;

    // adding substructure (PXB, PXF, TIB, TID, TOB, TEC, or DT, CSC, RPC,GEM) bits
    pattern |= (subdet & SubstrMask) << SubstrOffset;

    // adding layer/disk/wheel bits
    pattern |= (layer & LayerMask) << LayerOffset;

    // adding mono/stereo bit
    pattern |= (side & SideMask) << SideOffset;

    TrackingRecHit::Type patternHitType = (hitType == TrackingRecHit::missing_inner ||
                                           hitType == TrackingRecHit::missing_outer) ? TrackingRecHit::missing : hitType;

    pattern |= (patternHitType & HitTypeMask) << HitTypeOffset;

    return pattern;
}

bool HitPattern::appendHit(const TrackingRecHit &hit, const TrackerTopology& ttopo)
{
    return appendHit(hit.geographicalId(), hit.getType(), ttopo);
}

bool HitPattern::appendHit(const DetId &id, TrackingRecHit::Type hitType, const TrackerTopology& ttopo)
{
    //if HitPattern is full, journey ends no matter what.
    if unlikely((hitCount == HitPattern::MaxHits)) {
        return false;
    }

    uint16_t pattern = HitPattern::encode(id, hitType, ttopo);

    return appendHit(pattern, hitType);
}

bool HitPattern::appendHit(const uint16_t pattern, TrackingRecHit::Type hitType)
{
    //if HitPattern is full, journey ends no matter what.
    if unlikely((hitCount == HitPattern::MaxHits)) {
        return false;
    }

    switch (hitType) {
    case TrackingRecHit::valid:
    case TrackingRecHit::missing:
    case TrackingRecHit::inactive:
    case TrackingRecHit::bad:
        // hitCount != endT => we are not inserting T type of hits but of T'
        // 0 != beginT || 0 != endT => we already have hits of T type
        // so we already have hits of T in the vector and we don't want to
        // mess them with T' hits.
        if unlikely(((hitCount != endTrackHits) && (0 != beginTrackHits || 0 != endTrackHits))) {
            cms::Exception("HitPattern")
                    << "TRACK_HITS"
                    << " were stored on this object before hits of some other category were inserted "
                    << "but hits of the same category should be inserted in a row. "
                    << "Please rework the code so it inserts all "
                    << "TRACK_HITS"
                    << " in a row.";
            return false;
        }
        return insertTrackHit(pattern);
        break;
    case TrackingRecHit::missing_inner:
        if unlikely(((hitCount != endInner) && (0 != beginInner || 0 != endInner))) {
            cms::Exception("HitPattern")
                    << "MISSING_INNER_HITS"
                    << " were stored on this object before hits of some other category were inserted "
                    << "but hits of the same category should be inserted in a row. "
                    << "Please rework the code so it inserts all "
                    << "MISSING_INNER_HITS"
                    << " in a row.";
            return false;
        }
        return insertExpectedInnerHit(pattern);
        break;
    case TrackingRecHit::missing_outer:
        if unlikely(((hitCount != endOuter) && (0 != beginOuter || 0 != endOuter))) {
            cms::Exception("HitPattern")
                    << "MISSING_OUTER_HITS"
                    << " were stored on this object before hits of some other category were inserted "
                    << "but hits of the same category should be inserted in a row. "
                    << "Please rework the code so it inserts all "
                    << "MISSING_OUTER_HITS"
                    << " in a row.";
            return false;
        }
        return insertExpectedOuterHit(pattern);
        break;
    }

    return false;
}

bool HitPattern::appendTrackerHit(uint16_t subdet, uint16_t layer, uint16_t stereo, TrackingRecHit::Type hitType) {
    return appendHit(encode(DetId::Tracker, subdet, layer, stereo, hitType), hitType);
}

bool HitPattern::appendMuonHit(const DetId& id, TrackingRecHit::Type hitType) {
    //if HitPattern is full, journey ends no matter what.
    if unlikely((hitCount == HitPattern::MaxHits)) {
        return false;
    }

    if unlikely(id.det() != DetId::Muon) {
        throw cms::Exception("HitPattern") << "Got DetId from det " << id.det() << " that is not Muon in appendMuonHit(), which should only be used for muon hits in the HitPattern IO rule";
    }

    uint16_t detid = id.det();
    uint16_t subdet = id.subdetId();
    return appendHit(encode(detid, subdet, encodeMuonLayer(id), 0, hitType), hitType);
}

uint16_t HitPattern::getHitPatternByAbsoluteIndex(int position) const
{
    if unlikely((position < 0 || position >= hitCount)) {
        return HitPattern::EMPTY_PATTERN;
    }
    /*
    Note: you are not taking a consecutive sequence of HIT_LENGTH bits starting from position * HIT_LENGTH
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

     ugly enough, but it's not my fault, I was not even in CMS at that time   [gpetrucc] 
    */
  
    uint16_t bitEndOffset = (position + 1) * HIT_LENGTH;
    uint8_t secondWord   = (bitEndOffset >> 4);
    uint8_t secondWordBits = bitEndOffset & (16 - 1); // that is, bitEndOffset % 16
    if (secondWordBits >= HIT_LENGTH) { // full block is in this word
      uint8_t lowBitsToTrash = secondWordBits - HIT_LENGTH;
      uint16_t myResult = (hitPattern[secondWord] >> lowBitsToTrash) & ((1 << HIT_LENGTH) - 1);
      return myResult;
    } else {
      uint8_t  firstWordBits   = HIT_LENGTH - secondWordBits;
      uint16_t firstWordBlock  = hitPattern[secondWord - 1] >> (16 - firstWordBits);
      uint16_t secondWordBlock = hitPattern[secondWord] & ((1 << secondWordBits) - 1);
      uint16_t myResult = firstWordBlock + (secondWordBlock << firstWordBits);
      return myResult;
    }
}

bool HitPattern::hasValidHitInFirstPixelBarrel() const
{
    for (int i = beginTrackHits; i < endTrackHits; ++i) {
        uint16_t pattern = getHitPatternByAbsoluteIndex(i);
        if (pixelBarrelHitFilter(pattern) && (getLayer(pattern) == 1)
                && validHitFilter(pattern)) {
            return true;
        }
    }
    return false;
}

bool HitPattern::hasValidHitInFirstPixelEndcap() const
{
    for (int i = beginTrackHits; i < endTrackHits; ++i) {
        uint16_t pattern = getHitPatternByAbsoluteIndex(i);
        if (pixelEndcapHitFilter(pattern) && (getLayer(pattern) == 1)
                && validHitFilter(pattern)) {
            return true;
        }
    }
    return false;
}

int HitPattern::numberOfValidStripLayersWithMonoAndStereo(uint16_t stripdet, uint16_t layer) const
{
    bool hasMono[SubstrMask + 1][LayerMask + 1];
    bool hasStereo[SubstrMask + 1][LayerMask + 1];
    memset(hasMono, 0, sizeof(hasMono));
    memset(hasStereo, 0, sizeof(hasStereo));

    // mark which layers have mono/stereo hits
    for (int i = beginTrackHits; i < endTrackHits; ++i) {
        uint16_t pattern = getHitPatternByAbsoluteIndex(i);
        uint16_t subStructure = getSubStructure(pattern);

        if (validHitFilter(pattern) && stripHitFilter(pattern)) {
            if (stripdet != 0 && subStructure != stripdet) {
                continue;
            }

            if (layer != 0 && getSubSubStructure(pattern) != layer) {
                continue;
            }

            switch (getSide(pattern)) {
            case 0: // mono
                hasMono[subStructure][getLayer(pattern)] = true;
                break;
            case 1: // stereo
                hasStereo[subStructure][getLayer(pattern)] = true;
                break;
            default:
                ;
                break;
            }
        }
    }

    // count how many layers have mono and stereo hits
    int count = 0;
    for (int i = 0; i < SubstrMask + 1; ++i) {
        for (int j = 0; j < LayerMask + 1; ++j) {
            if (hasMono[i][j] && hasStereo[i][j]) {
                count++;
            }
        }
    }
    return count;
}

int HitPattern::numberOfValidStripLayersWithMonoAndStereo() const
{
   auto category = TRACK_HITS;
   std::bitset<128> side[2];
   std::pair<uint8_t, uint8_t> range = getCategoryIndexRange(category);
   for (int i = range.first; i < range.second; ++i) {
     auto pattern = getHitPatternByAbsoluteIndex(i);
     if (pattern<minStripWord) continue;
     uint16_t hitType = (pattern >> HitTypeOffset) & HitTypeMask;
     if (hitType != HIT_TYPE::VALID) continue;
     auto apattern = (pattern-minTrackerWord) >> LayerOffset;
     // assert(apattern<128);
     side[getSide(pattern)].set(apattern);
   }
   // assert(numberOfValidStripLayersWithMonoAndStereo(0, 0)==int((side[0]&side[1]).count()));
   return (side[0]&side[1]).count();


}

int HitPattern::numberOfValidTOBLayersWithMonoAndStereo(uint32_t layer) const
{
    return numberOfValidStripLayersWithMonoAndStereo(StripSubdetector::TOB, layer);
}

int HitPattern::numberOfValidTIBLayersWithMonoAndStereo(uint32_t layer) const
{
    return numberOfValidStripLayersWithMonoAndStereo(StripSubdetector::TIB, layer);
}

int HitPattern::numberOfValidTIDLayersWithMonoAndStereo(uint32_t layer) const
{
    return numberOfValidStripLayersWithMonoAndStereo(StripSubdetector::TID, layer);
}

int HitPattern::numberOfValidTECLayersWithMonoAndStereo(uint32_t layer) const
{
    return numberOfValidStripLayersWithMonoAndStereo(StripSubdetector::TEC, layer);
}

uint32_t HitPattern::getTrackerLayerCase(HitCategory category, uint16_t substr, uint16_t layer) const
{
    uint16_t tk_substr_layer = (0x1 << SubDetectorOffset)
                               + ((substr & SubstrMask) << SubstrOffset)
                               + ((layer & LayerMask) << LayerOffset);

    uint16_t mask = (SubDetectorMask << SubDetectorOffset)
                    + (SubstrMask << SubstrOffset)
                    + (LayerMask << LayerOffset);

    // layer case 0: valid + (missing, off, bad) ==> with measurement
    // layer case 1: missing + (off, bad) ==> without measurement
    // layer case 2: off, bad ==> totally off or bad, cannot say much
    // layer case NULL_RETURN: track outside acceptance or in gap ==> null
    uint32_t layerCase = NULL_RETURN;
    std::pair<uint8_t, uint8_t> range = getCategoryIndexRange(category);
    for (int i = range.first; i < range.second; ++i) {
        uint16_t pattern = getHitPatternByAbsoluteIndex(i);
        if ((pattern & mask) == tk_substr_layer) {
            uint16_t hitType = (pattern >> HitTypeOffset) & HitTypeMask;
            if (hitType < layerCase) {
                // BAD and INACTIVE as the same type (as INACTIVE)
                layerCase = (hitType == HIT_TYPE::BAD ? HIT_TYPE::INACTIVE : hitType);
                if (layerCase == HIT_TYPE::VALID) {
                    break;
                }
            }
        }
    }
    return layerCase;
}

uint16_t HitPattern::getTrackerMonoStereo(HitCategory category, uint16_t substr, uint16_t layer) const
{
    uint16_t tk_substr_layer = (0x1 << SubDetectorOffset)
                               + ((substr & SubstrMask) << SubstrOffset)
                               + ((layer & LayerMask) << LayerOffset);
    uint16_t mask = (SubDetectorMask << SubDetectorOffset)
                    + (SubstrMask << SubstrOffset)
                    + (LayerMask << LayerOffset);

    //             0: neither a valid mono nor a valid stereo hit
    //          MONO: valid mono hit
    //        STEREO: valid stereo hit
    // MONO | STEREO: both
    uint16_t monoStereo = 0x0;
    std::pair<uint8_t, uint8_t> range = getCategoryIndexRange(category);
    for (int i = range.first; i < range.second; ++i) {
        uint16_t pattern = getHitPatternByAbsoluteIndex(i);
        if ((pattern & mask) == tk_substr_layer) {
            uint16_t hitType = (pattern >> HitTypeOffset) & HitTypeMask;
            if (hitType == HIT_TYPE::VALID) {
                switch (getSide(pattern)) {
                case 0: // mono
                    monoStereo |= MONO;
                    break;
                case 1: // stereo
                    monoStereo |= STEREO;
                    break;
                }
            }

            if (monoStereo == (MONO | STEREO)) {
                break;
            }
        }
    }
    return monoStereo;
}


int HitPattern::pixelLayersWithMeasurement() const {
   auto category = TRACK_HITS;
   std::bitset<128> layerOk;
   std::pair<uint8_t, uint8_t> range = getCategoryIndexRange(category);
   for (int i = range.first; i < range.second; ++i) {
     auto pattern = getHitPatternByAbsoluteIndex(i);
     if unlikely(!trackerHitFilter(pattern)) continue;
     if (pattern>minStripWord) continue;
     uint16_t hitType = (pattern >> HitTypeOffset) & HitTypeMask;
     if (hitType != HIT_TYPE::VALID) continue;
     pattern = (pattern-minTrackerWord) >> LayerOffset;
     // assert(pattern<128);
     layerOk.set(pattern);
   }
   // assert(pixelLayersWithMeasurementOld()==int(layerOk.count()));
   return layerOk.count();
}


int HitPattern::trackerLayersWithMeasurement() const {
   auto category = TRACK_HITS;
   std::bitset<128> layerOk;
   std::pair<uint8_t, uint8_t> range = getCategoryIndexRange(category);
   for (int i = range.first; i < range.second; ++i) {
     auto pattern = getHitPatternByAbsoluteIndex(i);
     if unlikely(!trackerHitFilter(pattern)) continue;
     uint16_t hitType = (pattern >> HitTypeOffset) & HitTypeMask;
     if (hitType != HIT_TYPE::VALID) continue;
     pattern = (pattern-minTrackerWord) >> LayerOffset;
     // assert(pattern<128);
     layerOk.set(pattern);
   }
   // assert(trackerLayersWithMeasurementOld()==int(layerOk.count()));
   return layerOk.count(); 
}

int HitPattern::trackerLayersWithoutMeasurement(HitCategory category) const {
   std::bitset<128> layerOk;
   std::bitset<128> layerMissed;
   std::pair<uint8_t, uint8_t> range = getCategoryIndexRange(category);
   for (int i = range.first; i < range.second; ++i) {
     auto pattern = getHitPatternByAbsoluteIndex(i);
     if unlikely(!trackerHitFilter(pattern)) continue;
     uint16_t hitType = (pattern >> HitTypeOffset) & HitTypeMask;
     pattern = (pattern-minTrackerWord) >> LayerOffset;
     // assert(pattern<128);
     if (hitType == HIT_TYPE::VALID) layerOk.set(pattern);
     if (hitType == HIT_TYPE::MISSING) layerMissed.set(pattern);
   }
   layerMissed &= ~layerOk;

   // assert(trackerLayersWithoutMeasurementOld(category)==int(layerMissed.count()));

   return layerMissed.count();
 

}


int HitPattern::pixelBarrelLayersWithMeasurement() const
{
    int count = 0;
    uint16_t NPixBarrel = 4;
    for (uint16_t layer = 1; layer <= NPixBarrel; layer++) {
        if (getTrackerLayerCase(TRACK_HITS, PixelSubdetector::PixelBarrel, layer) == HIT_TYPE::VALID) {
            count++;
        }
    }
    return count;
}

int HitPattern::pixelEndcapLayersWithMeasurement() const
{
    int count = 0;
    uint16_t NPixForward = 3;
    for (uint16_t layer = 1; layer <= NPixForward; layer++) {
        if (getTrackerLayerCase(TRACK_HITS, PixelSubdetector::PixelEndcap, layer) == HIT_TYPE::VALID) {
            count++;
        }
    }
    return count;
}

int HitPattern::stripTIBLayersWithMeasurement() const
{
    int count = 0;
    for (uint16_t layer = 1; layer <= 4; layer++) {
        if (getTrackerLayerCase(TRACK_HITS, StripSubdetector::TIB, layer) == HIT_TYPE::VALID) {
            count++;
        }
    }
    return count;
}

int HitPattern::stripTIDLayersWithMeasurement() const
{
    int count = 0;
    for (uint16_t layer = 1; layer <= 3; layer++) {
        if (getTrackerLayerCase(TRACK_HITS, StripSubdetector::TID, layer) == HIT_TYPE::VALID) {
            count++;
        }
    }
    return count;
}

int HitPattern::stripTOBLayersWithMeasurement() const
{
    int count = 0;
    for (uint16_t layer = 1; layer <= 6; layer++) {
        if (getTrackerLayerCase(TRACK_HITS, StripSubdetector::TOB, layer) == HIT_TYPE::VALID) {
            count++;
        }
    }
    return count;
}

int HitPattern::stripTECLayersWithMeasurement() const
{
    int count = 0;
    for (uint16_t layer = 1; layer <= 9; layer++) {
        if (getTrackerLayerCase(TRACK_HITS, StripSubdetector::TEC, layer) == HIT_TYPE::VALID) {
            count++;
        }
    }
    return count;
}

int HitPattern::pixelBarrelLayersWithoutMeasurement(HitCategory category) const
{
    int count = 0;
    uint16_t NPixBarrel = 4;
    for (uint16_t layer = 1; layer <= NPixBarrel; layer++) {
        if (getTrackerLayerCase(category, PixelSubdetector::PixelBarrel, layer) == HIT_TYPE::MISSING) {
            count++;
        }
    }
    return count;
}

int HitPattern::pixelEndcapLayersWithoutMeasurement(HitCategory category) const
{
    int count = 0;
    uint16_t NPixForward = 3;
    for (uint16_t layer = 1; layer <= NPixForward; layer++) {
        if (getTrackerLayerCase(category, PixelSubdetector::PixelEndcap, layer) == HIT_TYPE::MISSING) {
            count++;
        }
    }
    return count;
}

int HitPattern::stripTIBLayersWithoutMeasurement(HitCategory category) const
{
    int count = 0;
    for (uint16_t layer = 1; layer <= 4; layer++) {
        if (getTrackerLayerCase(category, StripSubdetector::TIB, layer) == HIT_TYPE::MISSING) {
            count++;
        }
    }
    return count;
}

int HitPattern::stripTIDLayersWithoutMeasurement(HitCategory category) const
{
    int count = 0;
    for (uint16_t layer = 1; layer <= 3; layer++) {
        if (getTrackerLayerCase(category, StripSubdetector::TID, layer) == HIT_TYPE::MISSING) {
            count++;
        }
    }
    return count;
}

int HitPattern::stripTOBLayersWithoutMeasurement(HitCategory category) const
{
    int count = 0;
    for (uint16_t layer = 1; layer <= 6; layer++) {
        if (getTrackerLayerCase(category, StripSubdetector::TOB, layer) == HIT_TYPE::MISSING) {
            count++;
        }
    }
    return count;
}

int HitPattern::stripTECLayersWithoutMeasurement(HitCategory category) const
{
    int count = 0;
    for (uint16_t layer = 1; layer <= 9; layer++) {
        if (getTrackerLayerCase(category, StripSubdetector::TEC, layer) == HIT_TYPE::MISSING) {
            count++;
        }
    }
    return count;
}


int HitPattern::pixelBarrelLayersTotallyOffOrBad() const
{
    int count = 0;
    uint16_t NPixBarrel = 4;
    for (uint16_t layer = 1; layer <= NPixBarrel; layer++) {
        if (getTrackerLayerCase(TRACK_HITS, PixelSubdetector::PixelBarrel, layer) == HIT_TYPE::INACTIVE) {
            count++;
        }
    }
    return count;
}

int HitPattern::pixelEndcapLayersTotallyOffOrBad() const
{
    int count = 0;
    uint16_t NPixForward = 3;
    for (uint16_t layer = 1; layer <= NPixForward; layer++) {
        if (getTrackerLayerCase(TRACK_HITS, PixelSubdetector::PixelEndcap, layer) == HIT_TYPE::INACTIVE) {
            count++;
        }
    }
    return count;
}

int HitPattern::stripTIBLayersTotallyOffOrBad() const
{
    int count = 0;
    for (uint16_t layer = 1; layer <= 4; layer++) {
        if (getTrackerLayerCase(TRACK_HITS, StripSubdetector::TIB, layer) == HIT_TYPE::INACTIVE) {
            count++;
        }
    }
    return count;
}

int HitPattern::stripTIDLayersTotallyOffOrBad() const
{
    int count = 0;
    for (uint16_t layer = 1; layer <= 3; layer++) {
        if (getTrackerLayerCase(TRACK_HITS, StripSubdetector::TID, layer) == HIT_TYPE::INACTIVE) {
            count++;
        }
    }
    return count;
}

int HitPattern::stripTOBLayersTotallyOffOrBad() const
{
    int count = 0;
    for (uint16_t layer = 1; layer <= 6; layer++) {

        if (getTrackerLayerCase(TRACK_HITS, StripSubdetector::TOB, layer) == HIT_TYPE::INACTIVE) {
            count++;
        }
    }
    return count;
}

int HitPattern::stripTECLayersTotallyOffOrBad() const
{
    int count = 0;
    for (uint16_t layer = 1; layer <= 9; layer++) {
        if (getTrackerLayerCase(TRACK_HITS, StripSubdetector::TEC, layer) == HIT_TYPE::INACTIVE) {
            count++;
        }
    }
    return count;
}

int HitPattern::pixelBarrelLayersNull() const
{
    int count = 0;
    uint16_t NPixBarrel = 4;
    for (uint16_t layer = 1; layer <= NPixBarrel; layer++) {
        if (getTrackerLayerCase(TRACK_HITS, PixelSubdetector::PixelBarrel, layer) == NULL_RETURN) {
            count++;
        }
    }
    return count;
}

int HitPattern::pixelEndcapLayersNull() const
{
    int count = 0;
    uint16_t NPixForward = 3;
    for (uint16_t layer = 1; layer <= NPixForward; layer++) {
        if (getTrackerLayerCase(TRACK_HITS, PixelSubdetector::PixelEndcap, layer) == NULL_RETURN) {
            count++;
        }
    }
    return count;
}

int HitPattern::stripTIBLayersNull() const
{
    int count = 0;
    for (uint16_t layer = 1; layer <= 4; layer++) {
        if (getTrackerLayerCase(TRACK_HITS, StripSubdetector::TIB, layer) == NULL_RETURN) {
            count++;
        }
    }
    return count;
}

int HitPattern::stripTIDLayersNull() const
{
    int count = 0;
    for (uint16_t layer = 1; layer <= 3; layer++) {
        if (getTrackerLayerCase(TRACK_HITS, StripSubdetector::TID, layer) == NULL_RETURN) {
            count++;
        }
    }
    return count;
}

int HitPattern::stripTOBLayersNull() const
{
    int count = 0;
    for (uint16_t layer = 1; layer <= 6; layer++) {
        if (getTrackerLayerCase(TRACK_HITS, StripSubdetector::TOB, layer) == NULL_RETURN) {
            count++;
        }
    }
    return count;
}

int HitPattern::stripTECLayersNull() const
{
    int count = 0;
    for (uint16_t layer = 1; layer <= 9; layer++) {
        if (getTrackerLayerCase(TRACK_HITS, StripSubdetector::TEC, layer) == NULL_RETURN) {
            count++;
        }
    }
    return count;
}

void HitPattern::printHitPattern(HitCategory category, int position, std::ostream &stream) const
{
    uint16_t pattern = getHitPattern(category, position);
    stream << "\t";
    if (muonHitFilter(pattern)) {
        stream << "muon";
    } else if (trackerHitFilter(pattern)) {
        stream << "tracker";
    }

    stream << "\tsubstructure " << getSubStructure(pattern);
    if (muonHitFilter(pattern)) {
        stream << "\tstation " << getMuonStation(pattern);
        if (muonDTHitFilter(pattern)) {
            stream << "\tdt superlayer " << getDTSuperLayer(pattern);
        } else if (muonCSCHitFilter(pattern)) {
            stream << "\tcsc ring " << getCSCRing(pattern);
        } else if (muonRPCHitFilter(pattern)) {
            stream << "\trpc " << (getRPCregion(pattern) ? "endcaps" : "barrel")
                   << ", layer " << getRPCLayer(pattern);
       } else if (muonGEMHitFilter(pattern)) {
            stream << "\tgem " << (getGEMLayer(pattern) ? "layer1" : "layer2") 
                   << ", station " << getGEMStation(pattern);
        } else {
            stream << "(UNKNOWN Muon SubStructure!) \tsubsubstructure "
                   << getSubStructure(pattern);
        }
    } else {
        stream << "\tlayer " << getLayer(pattern);
    }
    stream << "\thit type " << getHitType(pattern);
    stream << std::endl;
}

void HitPattern::print(HitCategory category, std::ostream &stream) const
{
    stream << "HitPattern" << std::endl;
    for (int i = 0; i < numberOfHits(category); ++i) {
        printHitPattern(category, i, stream);
    }
    std::ios_base::fmtflags flags = stream.flags();
    stream.setf(std::ios_base::hex, std::ios_base::basefield);
    stream.setf(std::ios_base::showbase);

    for (int i = 0; i < this->numberOfHits(category); ++i) {
        stream << getHitPattern(category, i) << std::endl;
    }

    stream.flags(flags);
}

uint16_t HitPattern::isStereo(DetId i, const TrackerTopology& ttopo)
{
    if (i.det() != DetId::Tracker) {
        return 0;
    }

    switch (i.subdetId()) {
    case PixelSubdetector::PixelBarrel:
    case PixelSubdetector::PixelEndcap:
        return 0;
    case StripSubdetector::TIB:
        return ttopo.tibIsStereo(i);
    case StripSubdetector::TID:
        return ttopo.tidIsStereo(i);
    case StripSubdetector::TOB:
        return ttopo.tobIsStereo(i);
    case StripSubdetector::TEC:
        return ttopo.tecIsStereo(i);
    default:
        return 0;
    }
}

int HitPattern::muonStations(int subdet, int hitType) const
{
    int stations[4] = {0, 0, 0, 0};
    for (int i = beginTrackHits; i < endTrackHits; ++i) {
        uint16_t pattern = getHitPatternByAbsoluteIndex(i);
        if (muonHitFilter(pattern)
                && (subdet == 0   || int(getSubStructure(pattern)) == subdet)
                && (hitType == -1 || int(getHitType(pattern)) == hitType)) {
            stations[getMuonStation(pattern) - 1] = 1;
        }
    }

    return stations[0] + stations[1] + stations[2] + stations[3];
}

int HitPattern::innermostMuonStationWithHits(int hitType) const
{
    int ret = 0;
    for (int i = beginTrackHits; i < endTrackHits; ++i) {
        uint16_t pattern = getHitPatternByAbsoluteIndex(i);
        if (muonHitFilter(pattern)
                && (hitType == -1 || int(getHitType(pattern)) == hitType)) {
            int stat = getMuonStation(pattern);
            if (ret == 0 || stat < ret) {
                ret = stat;
            }
        }
    }

    return ret;
}

int HitPattern::outermostMuonStationWithHits(int hitType) const
{
    int ret = 0;
    for (int i = beginTrackHits; i < endTrackHits; ++i) {
        uint16_t pattern = getHitPatternByAbsoluteIndex(i);
        if (muonHitFilter(pattern) &&
                (hitType == -1 || int(getHitType(pattern)) == hitType)) {
            int stat = getMuonStation(pattern);
            if (ret == 0 || stat > ret) {
                ret = stat;
            }
        }
    }
    return ret;
}

int HitPattern::numberOfDTStationsWithRPhiView() const
{
    int stations[4] = {0, 0, 0, 0};
    for (int i = beginTrackHits; i < endTrackHits; ++i) {
        uint16_t pattern = getHitPatternByAbsoluteIndex(i);

        if (muonDTHitFilter(pattern) && validHitFilter(pattern)
                && getDTSuperLayer(pattern) != 2) {
            stations[getMuonStation(pattern) - 1] = 1;
        }
    }
    return stations[0] + stations[1] + stations[2] + stations[3];
}

int HitPattern::numberOfDTStationsWithRZView() const
{
    int stations[4] = {0, 0, 0, 0};
    for (int i = beginTrackHits; i < endTrackHits; ++i) {
        uint16_t pattern = getHitPatternByAbsoluteIndex(i);
        if (muonDTHitFilter(pattern) && validHitFilter(pattern)
                && getDTSuperLayer(pattern) == 2) {
            stations[getMuonStation(pattern) - 1] = 1;
        }
    }
    return stations[0] + stations[1] + stations[2] + stations[3];
}

int HitPattern::numberOfDTStationsWithBothViews() const
{
    int stations[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
    for (int i = beginTrackHits; i < endTrackHits; ++i) {
        uint16_t pattern = getHitPatternByAbsoluteIndex(i);
        if (muonDTHitFilter(pattern) && validHitFilter(pattern)) {
            stations[getMuonStation(pattern) - 1][getDTSuperLayer(pattern) == 2] = 1;
        }
    }

    return stations[0][0] * stations[0][1]
           + stations[1][0] * stations[1][1]
           + stations[2][0] * stations[2][1]
           + stations[3][0] * stations[3][1];
}

void HitPattern::insertHit(const uint16_t pattern)
{
    int offset = hitCount * HIT_LENGTH;
    for (int i = 0; i < HIT_LENGTH; i++) {
        int pos = offset + i;
        uint16_t bit = (pattern >> i) & 0x1;
        //equivalent to hitPattern[pos / 16] += bit << ((offset + i) % 16);
        hitPattern[pos >> 4] += bit << ((offset + i) & (16 - 1));
    }
    hitCount++;
}

bool HitPattern::insertTrackHit(const uint16_t pattern)
{
    // if begin is 0, this is the first hit of this type being inserted, so
    // we need to update begin so it points to the correct index, the first
    // empty index.
    // unlikely, because it will happen only when inserting
    // the first hit of this type
    if unlikely((0 == beginTrackHits && 0 == endTrackHits)) {
        beginTrackHits = hitCount;
        // before the first hit of this type is inserted, there are no hits
        endTrackHits = beginTrackHits;
    }

    insertHit(pattern);
    endTrackHits++;

    return true;
}

bool HitPattern::insertExpectedInnerHit(const uint16_t pattern)
{
    if unlikely((0 == beginInner && 0 == endInner)) {
        beginInner = hitCount;
        endInner = beginInner;
    }

    insertHit(pattern);
    endInner++;

    return true;
}

bool HitPattern::insertExpectedOuterHit(const uint16_t pattern)
{
    if unlikely((0 == beginOuter && 0 == endOuter)) {
        beginOuter = hitCount;
        endOuter = beginOuter;
    }

    insertHit(pattern);
    endOuter++;

    return true;
}

