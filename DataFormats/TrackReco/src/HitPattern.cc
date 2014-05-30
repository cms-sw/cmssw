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
using namespace std;

HitPattern::HitPattern() :
    hitCount(0),
    beginTrackHits(0),
    endTrackHits(0),
    beginInner(0),
    endInner(0),
    beginOuter(0),
    endOuter(0)
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
    endOuter(other.endOuter)
{
    memcpy(this->hitPattern, other.hitPattern, sizeof(uint16_t) * HitPattern::MaxHits);
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

    memset(this->hitPattern, EMPTY_PATTERN, sizeof(uint16_t) * HitPattern::MaxHits);
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

bool HitPattern::appendHit(const TrackingRecHitRef &ref)
{
    return appendHit(*ref);
}

uint16_t HitPattern::encode(const TrackingRecHit &hit)
{
    return encode(hit.geographicalId(), hit.getType());
}

uint16_t HitPattern::encode(const DetId &id, TrackingRecHit::Type hitType)
{
    uint16_t pattern = HitPattern::EMPTY_PATTERN;

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
            layer = ((DTLayerId(id.rawId()).station() - 1) << 2);
            layer |= DTLayerId(id.rawId()).superLayer();
            break;
        case MuonSubdetId::CSC:
            layer = ((CSCDetId(id.rawId()).station() - 1) << 2);
            layer |= (CSCDetId(id.rawId()).ring() - 1);
            break;
        case MuonSubdetId::RPC: {
            RPCDetId rpcid(id.rawId());
            layer = ((rpcid.station() - 1) << 2);
            layer |= (rpcid.station() <= 2) ? ((rpcid.layer() - 1) << 1) : 0x0;
            layer |= abs(rpcid.region());
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

    pattern |= (hitType & HitTypeMask) << HitTypeOffset;
    return pattern;
}

bool HitPattern::appendHit(const TrackingRecHit &hit)
{
    return appendHit(hit.geographicalId(), hit.getType());
}

bool HitPattern::appendHit(const DetId &id, TrackingRecHit::Type hitType)
{
    //if HitPattern is full, journey ends no matter what.
    if unlikely((hitCount == HitPattern::MaxHits)) {
        return false;
    }

    uint16_t pattern = HitPattern::encode(id, hitType);
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

uint16_t HitPattern::getHitPattern(HitCategory category, int position) const
{
    std::pair<uint8_t, uint8_t> range = getCategoryIndexRange(category);
    return ((position + range.first) < range.second) ? hitPattern[position + range.first] : HitPattern::EMPTY_PATTERN;
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

int HitPattern::numberOfValidTECLayersWithMonoAndStereo(HitCategory category, uint32_t layer) const
{
    return numberOfValidStripLayersWithMonoAndStereo(category, StripSubdetector::TEC, layer);
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
    // HitPattern is not full if we've reached this far.
    hitPattern[hitCount] = pattern;
    hitCount++;
    endTrackHits++;

    return true;
}

bool HitPattern::insertExpectedInnerHit(const uint16_t pattern)
{
    // Storing Hits has preference over storing expectedHits.
    // end == 0 means we haven't inserted any hits yet, but we still might,
    // so we neeed to check for reservations.

    if unlikely((0 == beginInner && 0 == endInner)) {
        beginInner = hitCount;
        endInner = beginInner;
    }

    hitPattern[hitCount] = pattern;
    hitCount++;
    endInner++;

    return true;
}

bool HitPattern::insertExpectedOuterHit(const uint16_t pattern)
{
    if unlikely((0 == beginOuter && 0 == endOuter)) {
        beginOuter = hitCount;
        endOuter = beginOuter;
    }

    hitPattern[hitCount] = pattern;
    hitCount++;
    endOuter++;

    return true;
}

