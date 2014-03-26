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
#include <iostream>

using namespace reco;
using namespace std;

HitPattern::HitPattern() :
    hitCount(0),
    beginTrackHits(0),
    endTrackHits(0),
    beginInner(0),
    endInner(0),
    beginOuter(0),
    endOuter(0),
    trackHitsCache(nullptr),
    expectedInnerHitsCache(nullptr),
    expectedOuterHitsCache(nullptr),
    trackHitsCacheDirty(true),
    expectedInnerHitsCacheDirty(false),
    expectedOuterHitsCacheDirty(true),
    defaultCategory(ALL_HITS)
{
    memset(hitPattern, HitPattern::EMPTY_PATTERN, sizeof(uint16_t) * HitPattern::MaxHits);
}

HitPattern::HitPattern(const HitPattern &other) :
    hitCount(other.hitCount),
    beginTrackHits(other.beginTrackHits),
    endTrackHits(other.endTrackHits),
    beginInner(other.beginInner),
    endInner(other.endInner),
    beginOuter(other.beginOuter),
    endOuter(other.endOuter),
    trackHitsCache(nullptr),
    expectedInnerHitsCache(nullptr),
    expectedOuterHitsCache(nullptr),
    trackHitsCacheDirty(true),
    expectedInnerHitsCacheDirty(true),
    expectedOuterHitsCacheDirty(true),
    defaultCategory(other.defaultCategory)
{
    memcpy(this->hitPattern, other.hitPattern, sizeof(uint16_t) * HitPattern::MaxHits);
}

HitPattern::~HitPattern()
{
    delete this->trackHitsCache;
    delete this->expectedOuterHitsCache;
    delete this->expectedInnerHitsCache;
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
    
    this->defaultCategory = other.defaultCategory;

    delete this->trackHitsCache;
    delete this->expectedInnerHitsCache;
    delete this->expectedOuterHitsCache;

    this->trackHitsCache = nullptr;
    this->expectedInnerHitsCache = nullptr;
    this->expectedOuterHitsCache = nullptr;

    this->trackHitsCacheDirty = true;
    this->expectedInnerHitsCacheDirty = true;
    this->expectedOuterHitsCacheDirty = true;

    memcpy(this->hitPattern, other.hitPattern, sizeof(uint16_t) * HitPattern::MaxHits);

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

    delete this->trackHitsCache;
    delete this->expectedOuterHitsCache;
    delete this->expectedInnerHitsCache;

    this->trackHitsCache = nullptr;
    this->expectedInnerHitsCache = nullptr;
    this->expectedOuterHitsCache = nullptr;

    this->trackHitsCacheDirty = true;
    this->expectedInnerHitsCacheDirty = true;
    this->expectedOuterHitsCacheDirty = true;

    this->defaultCategory = ALL_HITS;

    memset(this->hitPattern, EMPTY_PATTERN, sizeof(uint16_t) * HitPattern::MaxHits);
}

const HitPattern & HitPattern::getTrackHits() const
{
    if (defaultCategory == HitCategory::TRACK_HITS){
        return *this;
    }

    if (trackHitsCache == nullptr) {
        trackHitsCache = new HitPattern();
    }

    if (trackHitsCacheDirty) {
        trackHitsCacheDirty = false;
        *trackHitsCache = getHitsByCategory(HitCategory::TRACK_HITS);
    }

    return *trackHitsCache;
}

const HitPattern & HitPattern::getExpectedInnerHits() const
{
    if (defaultCategory == HitCategory::MISSING_INNER_HITS){
        return *this;
    }

    if (expectedInnerHitsCache == nullptr) {
        expectedInnerHitsCache = new HitPattern();
    }

    if (expectedInnerHitsCacheDirty) {
        expectedInnerHitsCacheDirty = false;
        *expectedInnerHitsCache = getHitsByCategory(HitCategory::MISSING_INNER_HITS);
    }

    return *expectedInnerHitsCache;
}

const HitPattern & HitPattern::getExpectedOuterHits() const
{
    if (defaultCategory == HitCategory::MISSING_OUTER_HITS){
        return *this;
    }

    if (expectedOuterHitsCache == nullptr) {
        expectedOuterHitsCache = new HitPattern();
    }

    if (expectedOuterHitsCacheDirty) {
        expectedOuterHitsCacheDirty = false;
        *expectedOuterHitsCache = getHitsByCategory(HitCategory::MISSING_OUTER_HITS);
    }

    return *expectedOuterHitsCache;
}

HitPattern HitPattern::getHitsByCategory(HitCategory category) const
{
    HitPattern hits;
/*
void HitPattern::updateCache(HitCategory category) const
{
    HitPattern *hits;
    switch(category){
        case TRACK_HITS:
            hits = trackHitsCache;
        break;
        case MISSING_INNER_HITS:
            hits = expectedInnerHitsCache;
        break;
        case MISSING_OUTER_HITS:
            hits = expectedOuterHitsCache;
        break;
        default:
            return;
    }
*/
    hits.defaultCategory = category;

    std::pair<uint8_t, uint8_t> range = getCategoryIndexRange(category);

    int hitsToCopy = range.second - range.first;
    memcpy(hits.hitPattern, &hitPattern[range.first], sizeof(uint16_t) * hitsToCopy);

    hits.hitCount = hitsToCopy;

    switch (category) {
    case ALL_HITS:
        //same as copying
        hits.beginTrackHits = this->beginTrackHits;
        hits.endTrackHits = this->endTrackHits;
        hits.beginInner = this->beginInner;
        hits.endInner = this->endInner;
        hits.beginOuter = this->beginOuter;
        hits.endOuter = this->endOuter;
        break;
    case TRACK_HITS:
        hits.endTrackHits = hitsToCopy;
        break;
    case MISSING_INNER_HITS:
        hits.endInner = hitsToCopy;
        break;
    case MISSING_OUTER_HITS:
        hits.endOuter = hitsToCopy;
        break;
    }

    return hits;
}

std::pair<uint8_t, uint8_t> HitPattern::getCategoryIndexRange(HitCategory category) const
{
    switch (category) {
    case ALL_HITS:
        return std::pair<uint8_t, uint8_t>(0, hitCount);
        break;
    case TRACK_HITS:
        return std::pair<uint8_t, uint8_t>(beginTrackHits, endTrackHits);
        break;
    case MISSING_INNER_HITS:
        return std::pair<uint8_t, uint8_t>(beginInner, endInner);
        break;
    case MISSING_OUTER_HITS:
        return std::pair<uint8_t, uint8_t>(beginOuter, endOuter);
        break;
    }
    return std::pair<uint8_t, uint8_t>(-1, -1);
}

int HitPattern::countHits(HitCategory category, filterType filter) const
{
    int count = 0;
    std::pair<uint8_t, uint8_t> range = getCategoryIndexRange(category);
    for (int i = range.first; i < range.second; ++i) {
        if (filter(getHitPatternByAbsoluteIndex(i))) {
            ++count;
        }
    }
    return count;
}

int HitPattern::countTypedHits(HitCategory category, filterType typeFilter, filterType filter) const
{
    int count = 0;
    std::pair<uint8_t, uint8_t> range = getCategoryIndexRange(category);
    for (int i = range.first; i < range.second; ++i) {
        uint16_t pattern = getHitPatternByAbsoluteIndex(i);
        if (typeFilter(pattern) && filter(pattern)) {
            ++count;
        }
    }
    return count;
}

bool HitPattern::insert(const TrackingRecHit &hit)
{
    //if HitPattern is full, journey ends no matter what.
    if unlikely((hitCount == HitPattern::MaxHits)) {
        return false;
    }

    uint16_t pattern = HitPattern::encode(hit);
    switch (hit.getType()) {
    case TrackingRecHit::valid:
    case TrackingRecHit::missing:
    case TrackingRecHit::inactive:
    case TrackingRecHit::bad:
        // hitCount != endT => we are not inserting T type of hits but of T'
        // 0 != beginT || 0 != endT => we already have hits of T type
        // so we already have hits of T in the vector and we don't want to
        // mess them with T' hits.
        if unlikely(((hitCount != endTrackHits) && (0 != beginTrackHits || 0 != endTrackHits))) {
            return false;
        }
        return insertTrackHit(pattern);
        break;
    case TrackingRecHit::missing_inner:
        if unlikely(((hitCount != endInner) && (0 != beginInner || 0 != endInner))) {
            return false;
        }
        return insertExpectedInnerHit(pattern);
        break;
    case TrackingRecHit::missing_outer:
        if unlikely(((hitCount != endOuter) && (0 != beginOuter || 0 != endOuter))) {
            return false;
        }
        return insertExpectedOuterHit(pattern);
        break;
    }

    return false;
}

bool HitPattern::insert(const TrackingRecHitRef &ref)
{
    return insert(*ref);
}

uint16_t HitPattern::encode(const TrackingRecHit &hit)
{
    uint16_t pattern = HitPattern::EMPTY_PATTERN;

    DetId id = hit.geographicalId();
    uint16_t detid = id.det();

    // adding tracker/muon detector bit
    pattern |= (detid & SubDetectorMask) << SubDetectorOffset;

    // adding substructure (PXB, PXF, TIB, TID, TOB, TEC, or DT, CSC, RPC) bits
    uint16_t subdet = id.subdetId();
    pattern |= (subdet & SubstrMask) << SubstrOffset;

    // adding layer/disk/wheel bits
    uint16_t layer = 0x0;
    if (detid == DetId::Tracker) {
        switch (subdet) {
        case PixelSubdetector::PixelBarrel:
            layer = PXBDetId(id).layer();
            break;
        case PixelSubdetector::PixelEndcap:
            layer = PXFDetId(id).disk();
            break;
        case StripSubdetector::TIB:
            layer = TIBDetId(id).layer();
            break;
        case StripSubdetector::TID:
            layer = TIDDetId(id).wheel();
            break;
        case StripSubdetector::TOB:
            layer = TOBDetId(id).layer();
            break;
        case StripSubdetector::TEC:
            layer = TECDetId(id).wheel();
            break;
        }
    } else if (detid == DetId::Muon) {
        switch (subdet) {
        case MuonSubdetId::DT:
            layer = ((DTLayerId(id.rawId()).station() - 1) << 2)
                    + DTLayerId(id.rawId()).superLayer();
            break;
        case MuonSubdetId::CSC:
            layer = ((CSCDetId(id.rawId()).station() - 1) << 2)
                    + (CSCDetId(id.rawId()).ring() - 1);
            break;
        case MuonSubdetId::RPC: 
        {
            RPCDetId rpcid(id.rawId());
            layer = ((rpcid.station() - 1) << 2) + abs(rpcid.region());
            if (rpcid.station() <= 2) {
                layer += 2 * (rpcid.layer() - 1);
            }
        }
        break;
        }
    }

    pattern |= (layer & LayerMask) << LayerOffset;

    // adding mono/stereo bit
    uint16_t side = 0x0;
    if (detid == DetId::Tracker) {
        side = isStereo(id);
    } else if (detid == DetId::Muon) {
        side = 0x0;
    }

    pattern |= (side & SideMask) << SideOffset;

    uint16_t hitType = (uint16_t) hit.getType();
    pattern |= (hitType & HitTypeMask) << HitTypeOffset;
    return pattern;
}

bool HitPattern::appendHit(const TrackingRecHit &hit)
{
    // get rec hit det id and rec hit type
    DetId id = hit.geographicalId();
    uint16_t detid = id.det();

    if (detid == DetId::Tracker) {
        return insert(hit);
    } else if (detid == DetId::Muon) {
        std::vector<const TrackingRecHit*> hits;
        uint16_t subdet = id.subdetId();
        if (subdet == (uint16_t) MuonSubdetId::DT) {
            if (hit.dimension() == 1) { // DT rechit (granularity 2)
                hits.push_back(&hit);
            } else if (hit.dimension() > 1) { // 4D segment (granularity 0).
                // Both 2 and 4 dim cases. MB4s have 2D, but formatted in 4D segment
                // 4D --> 2D
                std::vector<const TrackingRecHit*> seg2D = hit.recHits();
                // load 1D hits (2D --> 1D)
                for (std::vector<const TrackingRecHit*>::const_iterator it = seg2D.begin(); it != seg2D.end(); ++it) {
                    std::vector<const TrackingRecHit*> hits1D = (*it)->recHits();
                    copy(hits1D.begin(), hits1D.end(), back_inserter(hits));
                }
            }
        } else if (subdet == (uint16_t) MuonSubdetId::CSC) {
            if (hit.dimension() == 2) { // CSC rechit (granularity 2)
                hits.push_back(&hit);
            } else if (hit.dimension() == 4) { // 4D segment (granularity 0)
                // load 2D hits (4D --> 1D)
                hits = hit.recHits();
            }
        } else if (subdet == (uint16_t) MuonSubdetId::RPC) {
            hits.push_back(&hit);
        }

        bool allInserted = true;
        for (std::vector<const TrackingRecHit*>::const_iterator it = hits.begin(); it != hits.end(); ++it) {
            allInserted &= insert(**it);
            if unlikely((!allInserted)) {
                break;
            }
        }
        // notice it returns true only if all hits were inserted.
        return allInserted;
    }
    //neither tracker nor muon, nothing inserted so returning false makes sense;
    return false;
}

uint16_t HitPattern::getHitPattern(HitCategory category, int position) const
{
    std::pair<uint8_t, uint8_t> range = getCategoryIndexRange(category);
    return ((position + range.first) < range.second) ? hitPattern[position + range.first] : HitPattern::EMPTY_PATTERN;
}

uint16_t HitPattern::getHitPattern(int position) const
{
    return getHitPattern(defaultCategory, position);
}

uint16_t HitPattern::getHitPatternByAbsoluteIndex(int position) const
{
    return getHitPattern(ALL_HITS, position);
}

bool HitPattern::hasValidHitInFirstPixelBarrel(HitCategory category) const
{
    std::pair<uint8_t, uint8_t> range = getCategoryIndexRange(category);
    for (int i = range.first; i < range.second; ++i) {
        uint16_t pattern = getHitPatternByAbsoluteIndex(i);
        if (pixelBarrelHitFilter(pattern) && (getLayer(pattern) == 1)
                && validHitFilter(pattern)) {
            return true;
        }
    }
    return false;
}

bool HitPattern::hasValidHitInFirstPixelBarrel() const
{
    return hasValidHitInFirstPixelBarrel(defaultCategory);
}

bool HitPattern::hasValidHitInFirstPixelEndcap(HitCategory category) const
{
    std::pair<uint8_t, uint8_t> range = getCategoryIndexRange(category);
    for (int i = range.first; i < range.second; ++i) {
        uint16_t pattern = getHitPatternByAbsoluteIndex(i);
        if (pixelEndcapHitFilter(pattern) && (getLayer(pattern) == 1)
                && validHitFilter(pattern)) {
            return true;
        }
    }
    return false;
}

bool HitPattern::hasValidHitInFirstPixelEndcap() const
{
    return hasValidHitInFirstPixelEndcap(defaultCategory);
}

int HitPattern::numberOfHits(HitCategory category) const
{
    std::pair<uint8_t, uint8_t> range = getCategoryIndexRange(category);
    return range.second - range.first;
}

int HitPattern::numberOfValidStripLayersWithMonoAndStereo(HitCategory category, uint16_t stripdet, uint16_t layer) const
{
    bool hasMono[SubstrMask + 1][LayerMask + 1];
    bool hasStereo[SubstrMask + 1][LayerMask + 1];
    memset(hasMono, 0, sizeof(hasMono));
    memset(hasStereo, 0, sizeof(hasStereo));

    // mark which layers have mono/stereo hits
    std::pair<uint8_t, uint8_t> range = getCategoryIndexRange(category);
    for (int i = range.first; i < range.second; ++i) {
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

int HitPattern::numberOfValidStripLayersWithMonoAndStereo(HitCategory category) const
{
    return numberOfValidStripLayersWithMonoAndStereo(category, 0, 0);
}

int HitPattern::numberOfValidTOBLayersWithMonoAndStereo(HitCategory category, uint32_t layer) const
{
    return numberOfValidStripLayersWithMonoAndStereo(category, StripSubdetector::TOB, layer);
}

int HitPattern::numberOfValidTIBLayersWithMonoAndStereo(HitCategory category, uint32_t layer) const
{
    return numberOfValidStripLayersWithMonoAndStereo(category, StripSubdetector::TIB, layer);
}

int HitPattern::numberOfValidTIDLayersWithMonoAndStereo(HitCategory category, uint32_t layer) const
{
    return numberOfValidStripLayersWithMonoAndStereo(category, StripSubdetector::TID, layer);
}

int HitPattern::numberOfValidTIDLayersWithMonoAndStereo(uint32_t layer) const
{
    return numberOfValidTIDLayersWithMonoAndStereo(defaultCategory, layer);
}

int HitPattern::numberOfValidTECLayersWithMonoAndStereo(HitCategory category, uint32_t layer) const
{
    return numberOfValidStripLayersWithMonoAndStereo(category, StripSubdetector::TEC, layer);
}

int HitPattern::numberOfValidTECLayersWithMonoAndStereo(uint32_t layer) const
{
    return numberOfValidTECLayersWithMonoAndStereo(defaultCategory, layer);
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
                // treats BADS and MISSING as the same type
                layerCase = (hitType == HIT_TYPE::BAD ||
                             hitType == HIT_TYPE::MISSING_INNER ||
                             hitType == HIT_TYPE::MISSING_OUTER) 
                            ? HIT_TYPE::MISSING : hitType;
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

int HitPattern::pixelBarrelLayersWithMeasurement(HitCategory category) const
{
    int count = 0;
    uint16_t NPixBarrel = 4;
    for (uint16_t layer = 1; layer <= NPixBarrel; layer++) {
        if (getTrackerLayerCase(category, 
            PixelSubdetector::PixelBarrel, layer) == HIT_TYPE::VALID) {
            count++;
        }
    }
    return count;
}

int HitPattern::pixelEndcapLayersWithMeasurement(HitCategory category) const
{
    int count = 0;
    uint16_t NPixForward = 3;
    for (uint16_t layer = 1; layer <= NPixForward; layer++) {
        if (getTrackerLayerCase(category, PixelSubdetector::PixelEndcap, layer) == HIT_TYPE::VALID) {
            count++;
        }
    }
    return count;
}

int HitPattern::stripTIBLayersWithMeasurement(HitCategory category) const
{
    int count = 0;
    for (uint16_t layer = 1; layer <= 4; layer++) {
        if (getTrackerLayerCase(category, StripSubdetector::TIB, layer) == HIT_TYPE::VALID) {
            count++;
        }
    }
    return count;
}

int HitPattern::stripTIDLayersWithMeasurement(HitCategory category) const
{
    int count = 0;
    for (uint16_t layer = 1; layer <= 3; layer++) {
        if (getTrackerLayerCase(category, StripSubdetector::TID, layer) == HIT_TYPE::VALID) {
            count++;
        }
    }
    return count;
}

int HitPattern::stripTOBLayersWithMeasurement(HitCategory category) const
{
    int count = 0;
    for (uint16_t layer = 1; layer <= 6; layer++) {
        if (getTrackerLayerCase(category, StripSubdetector::TOB, layer) == HIT_TYPE::VALID) {
            count++;
        }
    }
    return count;
}

int HitPattern::stripTECLayersWithMeasurement(HitCategory category) const
{
    int count = 0;
    for (uint16_t layer = 1; layer <= 9; layer++) {
        if (getTrackerLayerCase(category, StripSubdetector::TEC, layer) == HIT_TYPE::VALID) {
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


int HitPattern::pixelBarrelLayersTotallyOffOrBad(HitCategory category) const
{
    int count = 0;
    uint16_t NPixBarrel = 4;
    for (uint16_t layer = 1; layer <= NPixBarrel; layer++) {
        if (getTrackerLayerCase(category, PixelSubdetector::PixelBarrel, layer) == HIT_TYPE::INACTIVE) {
            count++;
        }
    }
    return count;
}

int HitPattern::pixelEndcapLayersTotallyOffOrBad(HitCategory category) const
{
    int count = 0;
    uint16_t NPixForward = 3;
    for (uint16_t layer = 1; layer <= NPixForward; layer++) {
        if (getTrackerLayerCase(category, PixelSubdetector::PixelEndcap, layer) == HIT_TYPE::INACTIVE) {
            count++;
        }
    }
    return count;
}

int HitPattern::stripTIBLayersTotallyOffOrBad(HitCategory category) const
{
    int count = 0;
    for (uint16_t layer = 1; layer <= 4; layer++) {
        if (getTrackerLayerCase(category, StripSubdetector::TIB, layer) == HIT_TYPE::INACTIVE) {
            count++;
        }
    }
    return count;
}

int HitPattern::stripTIDLayersTotallyOffOrBad(HitCategory category) const
{
    int count = 0;
    for (uint16_t layer = 1; layer <= 3; layer++) {
        if (getTrackerLayerCase(category, StripSubdetector::TID, layer) == HIT_TYPE::INACTIVE) {
            count++;
        }
    }
    return count;
}

int HitPattern::stripTOBLayersTotallyOffOrBad(HitCategory category) const
{
    int count = 0;
    for (uint16_t layer = 1; layer <= 6; layer++) {
    
        if (getTrackerLayerCase(category, StripSubdetector::TOB, layer) == HIT_TYPE::INACTIVE) {
            count++;
        }
    }
    return count;
}

int HitPattern::stripTECLayersTotallyOffOrBad(HitCategory category) const
{
    int count = 0;
    for (uint16_t layer = 1; layer <= 9; layer++) {
        if (getTrackerLayerCase(category, StripSubdetector::TEC, layer) == HIT_TYPE::INACTIVE) {
            count++;
        }
    }
    return count;
}

int HitPattern::pixelBarrelLayersNull(HitCategory category) const
{
    int count = 0;
    uint16_t NPixBarrel = 4;
    for (uint16_t layer = 1; layer <= NPixBarrel; layer++) {
        if (getTrackerLayerCase(category, PixelSubdetector::PixelBarrel, layer) == NULL_RETURN) {
            count++;
        }
    }
    return count;
}

int HitPattern::pixelEndcapLayersNull(HitCategory category) const
{
    int count = 0;
    uint16_t NPixForward = 3;
    for (uint16_t layer = 1; layer <= NPixForward; layer++) {
        if (getTrackerLayerCase(category, PixelSubdetector::PixelEndcap, layer) == NULL_RETURN) {
            count++;
        }
    }
    return count;
}

int HitPattern::stripTIBLayersNull(HitCategory category) const
{
    int count = 0;
    for (uint16_t layer = 1; layer <= 4; layer++) {
        if (getTrackerLayerCase(category, StripSubdetector::TIB, layer) == NULL_RETURN) {
            count++;
        }
    }
    return count;
}

int HitPattern::stripTIDLayersNull(HitCategory category) const
{
    int count = 0;
    for (uint16_t layer = 1; layer <= 3; layer++) {
        if (getTrackerLayerCase(category, StripSubdetector::TID, layer) == NULL_RETURN) {
            count++;
        }
    }
    return count;
}

int HitPattern::stripTOBLayersNull(HitCategory category) const
{
    int count = 0;
    for (uint16_t layer = 1; layer <= 6; layer++) {
        if (getTrackerLayerCase(category, StripSubdetector::TOB, layer) == NULL_RETURN) {
            count++;
        }
    }
    return count;
}

int HitPattern::stripTECLayersNull(HitCategory category) const
{
    int count = 0;
    for (uint16_t layer = 1; layer <= 9; layer++) {
        if (getTrackerLayerCase(category, StripSubdetector::TEC, layer) == NULL_RETURN) {
            count++;
        }
    }
    return count;
}

void HitPattern::printHitPattern(int position, std::ostream &stream) const
{
    uint16_t pattern = getHitPatternByAbsoluteIndex(position);
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


void HitPattern::print(std::ostream &stream) const
{
    return print(defaultCategory, stream);
}

void HitPattern::print(HitCategory category, std::ostream &stream) const
{
    stream << "HitPattern" << std::endl;
    std::pair<uint8_t, uint8_t> range = getCategoryIndexRange(category);
    for (int i = range.first; i < range.second; ++i) {
        printHitPattern(i, stream);
    }
    std::ios_base::fmtflags flags = stream.flags();
    stream.setf(std::ios_base::hex, std::ios_base::basefield);
    stream.setf(std::ios_base::showbase);

//    for (int i = 0; i < this->    (); ++i) {
//        stream << getHitPatternByAbsoluteIndex(i) << std::endl;
//    }

    stream.flags(flags);
}

uint16_t HitPattern::isStereo(DetId i)
{
    if (i.det() != DetId::Tracker) {
        return 0;
    }

    switch (i.subdetId()) {
    case PixelSubdetector::PixelBarrel:
    case PixelSubdetector::PixelEndcap:
        return 0;
    case StripSubdetector::TIB: {
        TIBDetId id = i;
        return id.isStereo();
    }
    case StripSubdetector::TID: {
        TIDDetId id = i;
        return id.isStereo();
    }
    case StripSubdetector::TOB: {
        TOBDetId id = i;
        return id.isStereo();
    }
    case StripSubdetector::TEC: {
        TECDetId id = i;
        return id.isStereo();
    }
    default:
        return 0;
    }
}

int HitPattern::muonStations(HitCategory category, int subdet, int hitType) const
{
    int stations[4] = {0, 0, 0, 0};
    std::pair<uint8_t, uint8_t> range = getCategoryIndexRange(category);
    for (int i = range.first; i < range.second; ++i) {
        uint16_t pattern = getHitPatternByAbsoluteIndex(i);
        if (muonHitFilter(pattern)
                && (subdet == 0   || int(getSubStructure(pattern)) == subdet)
                && (hitType == -1 || int(getHitType(pattern)) == hitType)) {
            stations[getMuonStation(pattern) - 1] = 1;
        }
    }

    return stations[0] + stations[1] + stations[2] + stations[3];
}

int HitPattern::innermostMuonStationWithHits(HitCategory category, int hitType) const
{
    int ret = 0;
    std::pair<uint8_t, uint8_t> range = getCategoryIndexRange(category);
    for (int i = range.first; i < range.second; ++i) {
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

int HitPattern::outermostMuonStationWithHits(HitCategory category, int hitType) const
{
    int ret = 0;
    std::pair<uint8_t, uint8_t> range = getCategoryIndexRange(category);
    for (int i = range.first; i < range.second; ++i) {
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

int HitPattern::numberOfDTStationsWithRPhiView(HitCategory category) const
{
    int stations[4] = {0, 0, 0, 0};
    std::pair<uint8_t, uint8_t> range = getCategoryIndexRange(category);
    for (int i = range.first; i < range.second; ++i) {
        uint16_t pattern = getHitPatternByAbsoluteIndex(i);

        if (muonDTHitFilter(pattern) && validHitFilter(pattern)
                && getDTSuperLayer(pattern) != 2) {
            stations[getMuonStation(pattern) - 1] = 1;
        }
    }
    return stations[0] + stations[1] + stations[2] + stations[3];
}

int HitPattern::numberOfDTStationsWithRZView(HitCategory category) const
{
    int stations[4] = {0, 0, 0, 0};
    std::pair<uint8_t, uint8_t> range = getCategoryIndexRange(category);
    for (int i = range.first; i < range.second; ++i) {
        uint16_t pattern = getHitPatternByAbsoluteIndex(i);
        if (muonDTHitFilter(pattern) && validHitFilter(pattern)
                && getDTSuperLayer(pattern) == 2) {
            stations[getMuonStation(pattern) - 1] = 1;
        }
    }
    return stations[0] + stations[1] + stations[2] + stations[3];
}

int HitPattern::numberOfDTStationsWithBothViews(HitCategory category) const
{
    int stations[4][2] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
    std::pair<uint8_t, uint8_t> range = getCategoryIndexRange(category);
    for (int i = range.first; i < range.second; ++i) {
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

bool HitPattern::insertTrackHit(const uint16_t pattern)
{
    // if begin is 0, this is the first hit of this type being inserted, so
    // we need to update begin so it points to the correct index, the first
    // empty index.
    // unlikely, because it will happen only once, inserting
    // the first hit of this type
    if unlikely((0 == beginTrackHits && 0 == endTrackHits)) {
        beginTrackHits = hitCount;
        // before the first hit of this type is inserted, there are no hits
        endTrackHits = beginTrackHits;
    }

    // we know there is space available because trackHitsCache have preference and
    // HitPattern is not full if we reached this far.
    hitPattern[hitCount] = pattern;
    hitCount++;
    endTrackHits++;
    trackHitsCacheDirty = true;
    std::cout << std::endl << "HITS TOTALES " << (int)hitCount << std::endl;
    return true;
}

bool HitPattern::insertExpectedInnerHit(const uint16_t pattern)
{
    // Storing Hits has preference over storing expectedHits.
    // end == 0 means we haven't inserted any hits yet, but we still might,
    // so we neeed to check for reservations.
    if unlikely((0 == endTrackHits && ((HitPattern::MaxHits - hitCount) <= HitPattern::ReservedSpaceForHits))) {
        return false;
    }

    if unlikely((0 == beginInner && 0 == endInner)) {
        beginInner = hitCount;
        endInner = beginInner;
    }

    hitPattern[hitCount] = pattern;
    hitCount++;
    endInner++;
    expectedInnerHitsCacheDirty = true;
    std::cout << std::endl << "HITS TOTALES " << (int)hitCount << std::endl;
    return true;
}

bool HitPattern::insertExpectedOuterHit(const uint16_t pattern)
{
    if unlikely((0 == endTrackHits && ((HitPattern::MaxHits - hitCount) <= HitPattern::ReservedSpaceForHits))) {
        return false;
    }

    if unlikely((0 == beginOuter && 0 == endOuter)) {
        beginOuter = hitCount;
        endOuter = beginOuter;
    }

    hitPattern[hitCount] = pattern;
    hitCount++;
    endOuter++;
    expectedOuterHitsCacheDirty = true;
    std::cout << std::endl << "HITS TOTALES " << (int)hitCount << std::endl;
    return true;
}

int HitPattern::numberOfHits() const
{
    return numberOfHits(defaultCategory);
}

int HitPattern::numberOfTrackerHits() const
{
    return countHits(defaultCategory, trackerHitFilter);
}

int HitPattern::numberOfMuonHits() const
{
    return countHits(defaultCategory, muonHitFilter);
}

int HitPattern::numberOfValidHits() const
{
    return numberOfValidHits(defaultCategory);
}

int HitPattern::numberOfValidTrackerHits() const
{
    return numberOfValidTrackerHits(defaultCategory);
}

int HitPattern::numberOfValidMuonHits() const
{
    return numberOfValidMuonHits(defaultCategory);
}

int HitPattern::numberOfValidPixelHits() const
{
    return numberOfValidPixelHits(defaultCategory);
}

int HitPattern::numberOfValidPixelBarrelHits() const
{
    return numberOfValidPixelBarrelHits(defaultCategory);
}

int HitPattern::numberOfValidPixelEndcapHits() const
{
    return numberOfValidPixelEndcapHits(defaultCategory);
}

int HitPattern::numberOfValidStripHits() const
{
    return numberOfValidStripHits(defaultCategory);
}

int HitPattern::numberOfValidStripTIBHits() const
{
    return numberOfValidStripTIBHits(defaultCategory);
}

int HitPattern::numberOfValidStripTIDHits() const
{
    return numberOfValidStripTIDHits(defaultCategory);
}

int HitPattern::numberOfValidStripTOBHits() const
{
    return numberOfValidStripTOBHits(defaultCategory);
}

int HitPattern::numberOfValidStripTECHits() const
{
    return numberOfValidStripTECHits(defaultCategory);
}

int HitPattern::numberOfValidMuonDTHits() const
{
    return numberOfValidMuonDTHits(defaultCategory);
}

int HitPattern::numberOfValidMuonCSCHits() const
{
    return numberOfValidMuonCSCHits(defaultCategory);
}

int HitPattern::numberOfValidMuonRPCHits() const
{
    return numberOfValidMuonRPCHits(defaultCategory);
}

int HitPattern::numberOfLostHits() const
{
    return numberOfLostHits(defaultCategory);
}

int HitPattern::numberOfLostTrackerHits() const
{
    return numberOfLostTrackerHits(defaultCategory);
}

int HitPattern::numberOfLostMuonHits() const
{
    return numberOfLostMuonHits(defaultCategory);
}

int HitPattern::numberOfLostPixelHits() const
{
    return numberOfLostPixelHits(defaultCategory);
}

int HitPattern::numberOfLostPixelBarrelHits() const
{
    return numberOfLostPixelBarrelHits(defaultCategory);
}

int HitPattern::numberOfLostPixelEndcapHits() const
{
    return numberOfLostPixelEndcapHits(defaultCategory);
}

int HitPattern::numberOfLostStripHits() const
{
    return numberOfLostStripHits(defaultCategory);
}

int HitPattern::numberOfLostStripTIBHits() const
{
    return numberOfLostStripTIBHits(defaultCategory);
}

int HitPattern::numberOfLostStripTIDHits() const
{
    return numberOfLostStripTIDHits(defaultCategory);
}

int HitPattern::numberOfLostStripTOBHits() const
{
    return numberOfLostStripTOBHits(defaultCategory);
}

int HitPattern::numberOfLostStripTECHits() const
{
    return numberOfLostStripTECHits(defaultCategory);
}

int HitPattern::numberOfLostMuonDTHits() const
{
    return numberOfLostMuonDTHits(defaultCategory);
}

int HitPattern::numberOfLostMuonCSCHits() const
{
    return numberOfLostMuonCSCHits(defaultCategory);
}

int HitPattern::numberOfLostMuonRPCHits() const
{
    return numberOfLostMuonRPCHits(defaultCategory);
}

int HitPattern::numberOfBadHits() const
{
    return numberOfBadHits(defaultCategory);
}

int HitPattern::numberOfBadMuonHits() const
{
    return numberOfBadMuonHits(defaultCategory);
}

int HitPattern::numberOfBadMuonDTHits() const
{
    return numberOfBadMuonDTHits(defaultCategory);
}

int HitPattern::numberOfBadMuonCSCHits() const
{
    return numberOfBadMuonCSCHits(defaultCategory);
}

int HitPattern::numberOfBadMuonRPCHits() const
{
    return numberOfBadMuonRPCHits(defaultCategory);
}

int HitPattern::numberOfInactiveHits() const
{
    return numberOfInactiveHits(defaultCategory);
}

int HitPattern::numberOfInactiveTrackerHits() const
{
    return numberOfInactiveTrackerHits(defaultCategory);
}

int HitPattern::numberOfExpectedInnerHits() const
{
    return numberOfExpectedInnerHits(defaultCategory);
}

int HitPattern::numberOfExpectedOuterHits() const
{
    return numberOfExpectedOuterHits(defaultCategory);
}

int HitPattern::numberOfValidStripLayersWithMonoAndStereo(uint16_t stripdet, uint16_t layer) const
{
    return numberOfValidStripLayersWithMonoAndStereo(defaultCategory, stripdet, layer);
}

int HitPattern::numberOfValidStripLayersWithMonoAndStereo() const
{
    return numberOfValidStripLayersWithMonoAndStereo(defaultCategory);
}

int HitPattern::numberOfValidTOBLayersWithMonoAndStereo(uint32_t layer) const
{
    return numberOfValidTOBLayersWithMonoAndStereo(defaultCategory, layer);
}

int HitPattern::numberOfValidTIBLayersWithMonoAndStereo(uint32_t layer) const
{
    return numberOfValidTIBLayersWithMonoAndStereo(defaultCategory, layer);
}

uint32_t HitPattern::getTrackerLayerCase(uint16_t substr, uint16_t layer) const
{
    return getTrackerLayerCase(defaultCategory, substr, layer);
}

uint16_t HitPattern::getTrackerMonoStereo(uint16_t substr, uint16_t layer) const
{
    return getTrackerMonoStereo(defaultCategory, substr, layer);
}

int HitPattern::trackerLayersWithMeasurement() const
{
    return trackerLayersWithMeasurement(defaultCategory);
}

int HitPattern::pixelLayersWithMeasurement() const
{
    return pixelLayersWithMeasurement(defaultCategory);
}

int HitPattern::stripLayersWithMeasurement() const
{
    return stripLayersWithMeasurement(defaultCategory);
}

int HitPattern::pixelBarrelLayersWithMeasurement() const
{
    return pixelBarrelLayersWithMeasurement(defaultCategory);
}

int HitPattern::pixelEndcapLayersWithMeasurement() const
{
    return pixelEndcapLayersWithMeasurement(defaultCategory);
}

int HitPattern::stripTIBLayersWithMeasurement() const
{
    return stripTIBLayersWithMeasurement(defaultCategory);
}

int HitPattern::stripTIDLayersWithMeasurement() const
{
    return stripTIDLayersWithMeasurement(defaultCategory);
}

int HitPattern::stripTOBLayersWithMeasurement() const
{
    return stripTOBLayersWithMeasurement(defaultCategory);
}

int HitPattern::stripTECLayersWithMeasurement() const
{
    return stripTECLayersWithMeasurement(defaultCategory);
}

int HitPattern::trackerLayersWithoutMeasurement() const
{
    return trackerLayersWithoutMeasurement(defaultCategory);
}

int HitPattern::pixelLayersWithoutMeasurement() const
{
    return pixelLayersWithoutMeasurement(defaultCategory);
}

int HitPattern::stripLayersWithoutMeasurement() const
{
    return stripLayersWithoutMeasurement(defaultCategory);
}

int HitPattern::pixelBarrelLayersWithoutMeasurement() const
{
    return pixelBarrelLayersWithoutMeasurement(defaultCategory);
}

int HitPattern::pixelEndcapLayersWithoutMeasurement() const
{
    return pixelEndcapLayersWithoutMeasurement(defaultCategory);
}

int HitPattern::stripTIBLayersWithoutMeasurement() const
{
    return stripTIBLayersWithoutMeasurement(defaultCategory);
}

int HitPattern::stripTIDLayersWithoutMeasurement() const
{
    return stripTIDLayersWithoutMeasurement(defaultCategory);
}

int HitPattern::stripTOBLayersWithoutMeasurement() const
{
    return stripTOBLayersWithoutMeasurement(defaultCategory);
}

int HitPattern::stripTECLayersWithoutMeasurement() const
{
    return stripTECLayersWithoutMeasurement(defaultCategory);
}

int HitPattern::trackerLayersTotallyOffOrBad() const
{
    return trackerLayersTotallyOffOrBad(defaultCategory);
}

int HitPattern::pixelLayersTotallyOffOrBad() const
{
    return pixelLayersTotallyOffOrBad(defaultCategory);
}

int HitPattern::stripLayersTotallyOffOrBad() const
{
    return stripLayersTotallyOffOrBad(defaultCategory);
}

int HitPattern::pixelBarrelLayersTotallyOffOrBad() const
{
    return pixelBarrelLayersTotallyOffOrBad(defaultCategory);
}

int HitPattern::pixelEndcapLayersTotallyOffOrBad() const
{
    return pixelEndcapLayersTotallyOffOrBad(defaultCategory);
}

int HitPattern::stripTIBLayersTotallyOffOrBad() const
{
    return stripTIBLayersTotallyOffOrBad(defaultCategory);
}

int HitPattern::stripTIDLayersTotallyOffOrBad() const
{
    return stripTIDLayersTotallyOffOrBad(defaultCategory);
}

int HitPattern::stripTOBLayersTotallyOffOrBad() const
{
    return stripTOBLayersTotallyOffOrBad(defaultCategory);
}

int HitPattern::stripTECLayersTotallyOffOrBad() const
{
    return stripTECLayersTotallyOffOrBad(defaultCategory);
}

int HitPattern::trackerLayersNull() const
{
    return trackerLayersNull(defaultCategory);
}

int HitPattern::pixelLayersNull() const
{
    return pixelLayersNull(defaultCategory);
}

int HitPattern::stripLayersNull() const
{
    return stripLayersNull(defaultCategory);
}

int HitPattern::pixelBarrelLayersNull() const
{
    return pixelBarrelLayersNull(defaultCategory);
}

int HitPattern::pixelEndcapLayersNull() const
{
    return pixelEndcapLayersNull(defaultCategory);
}

int HitPattern::stripTIBLayersNull() const
{
    return stripTIBLayersNull(defaultCategory);
}

int HitPattern::stripTIDLayersNull() const
{
    return stripTIDLayersNull(defaultCategory);
}

int HitPattern::stripTOBLayersNull() const
{
    return stripTOBLayersNull(defaultCategory);
}

int HitPattern::stripTECLayersNull() const
{
    return stripTECLayersNull(defaultCategory);
}

int HitPattern::muonStations(int subdet, int hitType) const
{
    return muonStations(defaultCategory, subdet);
}

int HitPattern::muonStationsWithValidHits() const
{
    return muonStationsWithValidHits(defaultCategory);
}

int HitPattern::muonStationsWithBadHits() const
{
    return muonStationsWithBadHits(defaultCategory);
}

int HitPattern::muonStationsWithAnyHits() const
{
    return muonStationsWithAnyHits(defaultCategory);
}

int HitPattern::dtStationsWithValidHits() const
{
    return dtStationsWithValidHits(defaultCategory);
}

int HitPattern::dtStationsWithBadHits() const
{
    return dtStationsWithBadHits(defaultCategory);
}

int HitPattern::dtStationsWithAnyHits() const
{
    return dtStationsWithAnyHits(defaultCategory);
}

int HitPattern::cscStationsWithValidHits() const
{
    return cscStationsWithValidHits(defaultCategory);
}

int HitPattern::cscStationsWithBadHits() const
{
    return cscStationsWithBadHits(defaultCategory);
}

int HitPattern::cscStationsWithAnyHits() const
{
    return cscStationsWithAnyHits(defaultCategory);
}

int HitPattern::rpcStationsWithValidHits() const
{
    return rpcStationsWithValidHits(defaultCategory);
}

int HitPattern::rpcStationsWithBadHits() const
{
    return rpcStationsWithBadHits(defaultCategory);
}

int HitPattern::rpcStationsWithAnyHits() const
{
    return rpcStationsWithAnyHits(defaultCategory);
}

int HitPattern::innermostMuonStationWithHits(int hitType) const
{
    return innermostMuonStationWithHits(defaultCategory, hitType);
}

int HitPattern::innermostMuonStationWithValidHits() const
{
    return innermostMuonStationWithValidHits(defaultCategory);
}

int HitPattern::innermostMuonStationWithBadHits() const
{
    return innermostMuonStationWithBadHits(defaultCategory);
}

int HitPattern::innermostMuonStationWithAnyHits() const
{
    return innermostMuonStationWithAnyHits(defaultCategory);
}

int HitPattern::outermostMuonStationWithHits(int hitType) const
{
    return outermostMuonStationWithHits(hitType);
}

int HitPattern::outermostMuonStationWithValidHits() const
{
    return outermostMuonStationWithValidHits(defaultCategory);
}

int HitPattern::outermostMuonStationWithBadHits() const
{
    return outermostMuonStationWithBadHits(defaultCategory);
}

int HitPattern::outermostMuonStationWithAnyHits() const
{
    return outermostMuonStationWithAnyHits(defaultCategory);
}

int HitPattern::numberOfDTStationsWithRPhiView() const
{
    return numberOfDTStationsWithRPhiView(defaultCategory);
}

int HitPattern::numberOfDTStationsWithRZView() const
{
    return numberOfDTStationsWithRZView(defaultCategory);
}

int HitPattern::numberOfDTStationsWithBothViews() const
{
    return numberOfDTStationsWithBothViews(defaultCategory);
}
