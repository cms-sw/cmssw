// -*- C++ -*-

#ifndef TrackReco_HitPattern_h
#define TrackReco_HitPattern_h

//
// File: DataFormats/TrackReco/interface/HitPattern.h
//
// Marcel Vos, INFN Pisa
// v1.10 2007/05/08 bellan
// Zongru Wan, Kansas State University
// Jean-Roch Vlimant
// Kevin Burkett
// Boris Mangano
//
// Hit pattern is the summary information of the hits associated to track in
// AOD.  When RecHits are no longer available, the compact hit pattern should
// allow basic track selection based on the hits in various subdetectors.
// The hits of a track are saved in unit16_t hitPattern[MaxHits].
//
//                                            uint16_t
// +------------+--------+---------------+---------------------------+-----------------+----------------+
// |   padding  |  tk/mu | sub-structure |     sub-sub-structure     |     stereo      |    hit type    |
// +------------+--------+---------------+---------------------------+-----------------+----------------+
// | 15 14 13 12|   11   | 10   9    8   | 7      6     5     4      |        3        |   2    1    0  |  bit
// +------------+--------+---------------+---------------------------+-----------------+----------------|
// | 0  0  0  0 | tk = 1 |    PXB = 1    | layer = 1-3               |                 | hit type = 0-5 |
// | 0  0  0  0 | tk = 1 |    PXF = 2    | disk  = 1-2               |                 | hit type = 0-5 |
// | 0  0  0  0 | tk = 1 |    TIB = 3    | layer = 1-4               | 0=rphi,1=stereo | hit type = 0-5 |
// | 0  0  0  0 | tk = 1 |    TID = 4    | wheel = 1-3               | 0=rphi,1=stereo | hit type = 0-5 |
// | 0  0  0  0 | tk = 1 |    TOB = 5    | layer = 1-6               | 0=rphi,1=stereo | hit type = 0-5 |
// | 0  0  0  0 | tk = 1 |    TEC = 6    | wheel = 1-9               | 0=rphi,1=stereo | hit type = 0-5 |
// | 0  0  0  0 | mu = 0 |    DT  = 1    | 4*(stat-1)+superlayer     |                 | hit type = 0-3 |
// | 0  0  0  0 | mu = 0 |    CSC = 2    | 4*(stat-1)+(ring-1)       |                 | hit type = 0-3 |
// | 0  0  0  0 | mu = 0 |    RPC = 3    | 4*(stat-1)+2*layer+region |                 | hit type = 0-3 |
// +------------+--------+---------------+---------------------------+-----------------+----------------+
//
//  hit type, see DataFormats/TrackingRecHit/interface/TrackingRecHit.h
//      valid    = valid hit                                     = 0
//      missing  = detector is good, but no rec hit found        = 1
//      inactive = detector is off, so there was no hope         = 2
//      bad      = there were many bad strips within the ellipse = 3
//      missingInner = 4
//      missingOuter = 5
//
//  padding: Padding content does not matter, but I will strongly encourage
//           you to keep then under control and set to ZERO.

// It had been shown by Zongru using a 100 GeV muon sample with 5000 events
// uniform in eta and phi, the average (maximum) number of tracker hits is
// 13 (17) and the average (maximum) number of muon detector hits is about
// 26 (50). If the number of hits of a track is larger than 80 then the extra
// hits are ignored by hit pattern. The static hit pattern array might be
// improved to a dynamic one in the future.
//
// Because of tracking with/without overlaps and with/without hit-splitting,
// the final number of hits per track is pretty "variable". Compared with the
// number of valid hits, the number of crossed layers with measurement should
// be more robust to discriminate between good and fake track.
//
// Since 4-bit for sub-sub-structure is not enough to specify a muon layer,
// the layer case counting methods are implemented for tracker only. This is
// different from the hit counting methods which are implemented for both
// tracker and muon detector.
//
// Given a tracker layer, specified by sub-structure and layer, the method
// getTrackerLayerCase(substr, layer) groups all of the hits in the hit pattern
// array for the layer together and returns one of the four cases
//
//     crossed
//        layer case 0: valid + (missing, off, bad) ==> with measurement
//        layer case 1: missing + (off, bad) ==> without measurement
//        layer case 2: off, bad ==> totally off or bad, cannot say much
//     not crossed
//        layer case 999999: track outside acceptance or in gap ==> null
//
// Given a tracker layer, specified by sub-structure and layer, the method
// getTrackerMonoStereo(substr, layer) groups all of the valid hits in the hit
// pattern array for the layer together and returns
//
//              0: neither a valid mono nor a valid stereo hit
//           MONO: valid mono hit
//         STEREO: valid stereo hit
//  MONO | STEREO: both
//

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "FWCore/Utilities/interface/GCC11Compatibility.h"

#include <utility>
#include <algorithm>
#include <iostream>
#include <ostream>
#include <memory>

namespace reco
{

class HitPattern
{

public:
    enum {
        MONO = 1,
        STEREO = 2
    };

    enum HIT_TYPE {
        VALID = 0,
        MISSING = 1,
        INACTIVE = 2,
        BAD = 3,
        MISSING_INNER = 4,
        MISSING_OUTER = 5
    };

    enum HitCategory {
        ALL_HITS = 0,
        TRACK_HITS = 1,
        MISSING_INNER_HITS = 2,
        MISSING_OUTER_HITS = 3
    };

    static const unsigned short MaxHits = 72;
    static const uint32_t NULL_RETURN = 999999;
    static const uint16_t EMPTY_PATTERN = 0x0;

    static bool trackerHitFilter(uint16_t pattern);
    static bool muonHitFilter(uint16_t pattern);

    static bool validHitFilter(uint16_t pattern);
    static bool missingHitFilter(uint16_t pattern);
    static bool inactiveHitFilter(uint16_t pattern);
    static bool badHitFilter(uint16_t pattern);

    static bool expectedInnerHitFilter(uint16_t pattern);
    static bool expectedOuterHitFilter(uint16_t pattern);

    static bool pixelHitFilter(uint16_t pattern);
    static bool pixelBarrelHitFilter(uint16_t pattern);
    static bool pixelEndcapHitFilter(uint16_t pattern);
    static bool stripHitFilter(uint16_t pattern);
    static bool stripTIBHitFilter(uint16_t pattern);
    static bool stripTIDHitFilter(uint16_t pattern);
    static bool stripTOBHitFilter(uint16_t pattern);
    static bool stripTECHitFilter(uint16_t pattern);

    static bool muonDTHitFilter(uint16_t pattern);
    static bool muonCSCHitFilter(uint16_t pattern);
    static bool muonRPCHitFilter(uint16_t pattern);

    static uint32_t getHitType(uint16_t pattern);

    // mono (0) or stereo (1)
    static uint32_t getSide(uint16_t pattern);
    static uint32_t getLayer(uint16_t pattern);
    static uint32_t getSubSubStructure(uint16_t pattern);
    static uint32_t getSubStructure(uint16_t pattern);
    static uint32_t getSubDetector(uint16_t pattern);

    // Muon station (1-4). Only valid for muon patterns, of course.
    // only for patterns from muon, of course
    static uint16_t getMuonStation(uint16_t pattern);

    // DT superlayer (1-3). Where the "hit" was a DT segment, superlayer is 0.
    // Only valid for muon DT patterns, of course.
    static uint16_t getDTSuperLayer(uint16_t pattern); // only for DT patterns

    /// CSC ring (1-4). Only valid for muon CSC patterns, of course.
    static uint16_t getCSCRing(uint16_t pattern) ;

    /// RPC layer: for station 1 and 2, layer = 1(inner) or 2(outer);
    // for station 3, 4 layer is always 0. Only valid for muon RPC patterns, of course.
    static uint16_t getRPCLayer(uint16_t pattern) ;

    /// RPC region: 0 = barrel, 1 = endcap. Only valid for muon RPC patterns, of course.
    static uint16_t getRPCregion(uint16_t pattern);

    HitPattern();

    ~HitPattern();

    HitPattern(const HitPattern &other);

    HitPattern &operator=(const HitPattern &other);

    template<typename I>
    bool appendHits(const I &begin, const I &end);
    bool appendHit(const TrackingRecHit &hit);
    bool appendHit(const TrackingRecHitRef &ref);
    bool appendHit(const DetId &id, TrackingRecHit::Type hitType);

    // get the pattern of the position-th hit
    uint16_t getHitPattern(HitCategory category, int position) const;

    void clear();

    // print the pattern of the position-th hit
    void printHitPattern(HitCategory category, int position, std::ostream &stream) const;
    void print(HitCategory category, std::ostream &stream = std::cout) const;

    bool hasValidHitInFirstPixelBarrel() const; // has valid hit in PXB layer 1
    bool hasValidHitInFirstPixelEndcap() const; // has valid hit in PXF layer 1

    int numberOfHits(HitCategory category) const;                 // not-null

    int numberOfTrackerHits(HitCategory category) const;          // not-null, tracker
    int numberOfMuonHits() const;                                 // not-null, muon

    int numberOfValidHits() const;                                // not-null, valid
    int numberOfValidTrackerHits() const;                         // not-null, valid, tracker
    int numberOfValidMuonHits() const;                            // not-null, valid, muon

    int numberOfValidPixelHits() const;       // not-null, valid, pixel
    int numberOfValidPixelBarrelHits() const; // not-null, valid, pixel PXB
    int numberOfValidPixelEndcapHits() const; // not-null, valid, pixel PXF
    int numberOfValidStripHits() const;       // not-null, valid, strip
    int numberOfValidStripTIBHits() const;    // not-null, valid, strip TIB
    int numberOfValidStripTIDHits() const;    // not-null, valid, strip TID
    int numberOfValidStripTOBHits() const;    // not-null, valid, strip TOB
    int numberOfValidStripTECHits() const;    // not-null, valid, strip TEC
    int numberOfValidMuonDTHits() const;      // not-null, valid, muon DT
    int numberOfValidMuonCSCHits() const;     // not-null, valid, muon CSC
    int numberOfValidMuonRPCHits() const;     // not-null, valid, muon RPC

    int numberOfLostHits(HitCategory category) const;             // not-null, not valid
    int numberOfLostTrackerHits(HitCategory category) const;      // not-null, not valid, tracker
    int numberOfLostMuonHits() const;                             // not-null, not valid, muon
    int numberOfLostPixelHits(HitCategory category) const;        // not-null, not valid, pixel
    int numberOfLostPixelBarrelHits(HitCategory category) const;  // not-null, not valid, pixel PXB
    int numberOfLostPixelEndcapHits(HitCategory category) const;  // not-null, not valid, pixel PXF
    int numberOfLostStripHits(HitCategory category) const;        // not-null, not valid, strip
    int numberOfLostStripTIBHits(HitCategory category) const;     // not-null, not valid, strip TIB
    int numberOfLostStripTIDHits(HitCategory category) const;     // not-null, not valid, strip TID
    int numberOfLostStripTOBHits(HitCategory category) const;     // not-null, not valid, strip TOB
    int numberOfLostStripTECHits(HitCategory category) const;     // not-null, not valid, strip TEC
    int numberOfLostMuonDTHits() const;       // not-null, not valid, muon DT
    int numberOfLostMuonCSCHits() const;      // not-null, not valid, muon CSC
    int numberOfLostMuonRPCHits() const;      // not-null, not valid, muon RPC

    int numberOfBadHits() const;              // not-null, bad (only used in Muon Ch.)
    int numberOfBadMuonHits() const;          // not-null, bad, muon
    int numberOfBadMuonDTHits() const;        // not-null, bad, muon DT
    int numberOfBadMuonCSCHits() const;       // not-null, bad, muon CSC
    int numberOfBadMuonRPCHits() const;       // not-null, bad, muon RPC

    int numberOfInactiveHits(HitCategory category) const;         // not-null, inactive
    int numberOfInactiveTrackerHits(HitCategory category) const;  // not-null, inactive, tracker

    int numberOfExpectedInnerHits(HitCategory category) const;
    int numberOfExpectedOuterHits(HitCategory category) const;

    // count strip layers that have non-null, valid mono and stereo hits
    int numberOfValidStripLayersWithMonoAndStereo(HitCategory category, uint16_t stripdet, uint16_t layer) const;
    int numberOfValidStripLayersWithMonoAndStereo(HitCategory category) const;
    int numberOfValidTOBLayersWithMonoAndStereo(HitCategory category, uint32_t layer = 0) const;
    int numberOfValidTIBLayersWithMonoAndStereo(HitCategory category, uint32_t layer = 0) const;
    int numberOfValidTIDLayersWithMonoAndStereo(HitCategory category, uint32_t layer = 0) const;
    int numberOfValidTECLayersWithMonoAndStereo(HitCategory category, uint32_t layer = 0) const;

    uint32_t getTrackerLayerCase(HitCategory category, uint16_t substr, uint16_t layer) const;
    uint16_t getTrackerMonoStereo(HitCategory category, uint16_t substr, uint16_t layer) const;

    int trackerLayersWithMeasurement(HitCategory category) const;        // case 0: tracker
    int pixelLayersWithMeasurement(HitCategory category) const;          // case 0: pixel
    int stripLayersWithMeasurement(HitCategory category) const;          // case 0: strip
    int pixelBarrelLayersWithMeasurement(HitCategory category) const;    // case 0: pixel PXB
    int pixelEndcapLayersWithMeasurement(HitCategory category) const;    // case 0: pixel PXF
    int stripTIBLayersWithMeasurement(HitCategory category) const;       // case 0: strip TIB
    int stripTIDLayersWithMeasurement(HitCategory category) const;       // case 0: strip TID
    int stripTOBLayersWithMeasurement(HitCategory category) const;       // case 0: strip TOB
    int stripTECLayersWithMeasurement(HitCategory category) const;       // case 0: strip TEC

    int trackerLayersWithoutMeasurement(HitCategory category) const;     // case 1: tracker
    int pixelLayersWithoutMeasurement(HitCategory category) const;       // case 1: pixel
    int stripLayersWithoutMeasurement(HitCategory category) const;       // case 1: strip
    int pixelBarrelLayersWithoutMeasurement(HitCategory category) const; // case 1: pixel PXB
    int pixelEndcapLayersWithoutMeasurement(HitCategory category) const; // case 1: pixel PXF
    int stripTIBLayersWithoutMeasurement(HitCategory category) const;    // case 1: strip TIB
    int stripTIDLayersWithoutMeasurement(HitCategory category) const;    // case 1: strip TID
    int stripTOBLayersWithoutMeasurement(HitCategory category) const;    // case 1: strip TOB
    int stripTECLayersWithoutMeasurement(HitCategory category) const;    // case 1: strip TEC

    int trackerLayersTotallyOffOrBad(HitCategory category) const;        // case 2: tracker
    int pixelLayersTotallyOffOrBad(HitCategory category) const;          // case 2: pixel
    int stripLayersTotallyOffOrBad(HitCategory category) const;          // case 2: strip
    int pixelBarrelLayersTotallyOffOrBad(HitCategory category) const;    // case 2: pixel PXB
    int pixelEndcapLayersTotallyOffOrBad(HitCategory category) const;    // case 2: pixel PXF
    int stripTIBLayersTotallyOffOrBad(HitCategory category) const;       // case 2: strip TIB
    int stripTIDLayersTotallyOffOrBad(HitCategory category) const;       // case 2: strip TID
    int stripTOBLayersTotallyOffOrBad(HitCategory category) const;       // case 2: strip TOB
    int stripTECLayersTotallyOffOrBad(HitCategory category) const;       // case 2: strip TEC

    int trackerLayersNull(HitCategory category) const;                   // case NULL_RETURN: tracker
    int pixelLayersNull(HitCategory category) const;                     // case NULL_RETURN: pixel
    int stripLayersNull(HitCategory category) const;                     // case NULL_RETURN: strip
    int pixelBarrelLayersNull(HitCategory category) const;               // case NULL_RETURN: pixel PXB
    int pixelEndcapLayersNull(HitCategory category) const;               // case NULL_RETURN: pixel PXF
    int stripTIBLayersNull(HitCategory category) const;                  // case NULL_RETURN: strip TIB
    int stripTIDLayersNull(HitCategory category) const;                  // case NULL_RETURN: strip TID
    int stripTOBLayersNull(HitCategory category) const;                  // case NULL_RETURN: strip TOB
    int stripTECLayersNull(HitCategory category) const;                  // case NULL_RETURN: strip TEC

    /// subdet = 0(all), 1(DT), 2(CSC), 3(RPC); hitType=-1(all), 0=valid, 3=bad
    int muonStations(int subdet, int hitType) const ;

    int muonStationsWithValidHits() const;
    int muonStationsWithBadHits() const;
    int muonStationsWithAnyHits() const;

    int dtStationsWithValidHits() const;
    int dtStationsWithBadHits() const;
    int dtStationsWithAnyHits() const;

    int cscStationsWithValidHits() const;
    int cscStationsWithBadHits() const;
    int cscStationsWithAnyHits() const;

    int rpcStationsWithValidHits() const;
    int rpcStationsWithBadHits() const;
    int rpcStationsWithAnyHits() const;

    /// hitType=-1(all), 0=valid, 3=bad; 0 = no stations at all
    int innermostMuonStationWithHits(int hitType) const;
    int innermostMuonStationWithValidHits() const;
    int innermostMuonStationWithBadHits() const;
    int innermostMuonStationWithAnyHits() const;

    /// hitType=-1(all), 0=valid, 3=bad; 0 = no stations at all
    int outermostMuonStationWithHits(int hitType) const;
    int outermostMuonStationWithValidHits() const;
    int outermostMuonStationWithBadHits() const;
    int outermostMuonStationWithAnyHits() const;

    int numberOfDTStationsWithRPhiView() const;
    int numberOfDTStationsWithRZView() const;
    int numberOfDTStationsWithBothViews() const;

private:
    // 3 bits for hit type
    const static unsigned short HitTypeOffset = 0;
    const static unsigned short HitTypeMask = 0x7;

    // 1 bit to identify the side in double-sided detectors
    const static unsigned short SideOffset = 3;
    const static unsigned short SideMask = 0x1;

    // 4 bits to identify the layer/disk/wheel within the substructure
    const static unsigned short LayerOffset = 4;
    const static unsigned short LayerMask = 0xF;

    // 3 bits to identify the tracker/muon detector substructure
    const static unsigned short SubstrOffset = 8;
    const static unsigned short SubstrMask = 0x7;

    // 1 bit to distinguish tracker and muon subsystems
    const static unsigned short SubDetectorOffset = 11;
    const static unsigned short SubDetectorMask = 0x1;

    //////////////////////////////////////////////////////
    // unused bits start at offset 12 -> bits [12, 15]. //
    //////////////////////////////////////////////////////

    // detector side for tracker modules (mono/stereo)
    static uint16_t isStereo(DetId i);
    static bool stripSubdetectorHitFilter(uint16_t pattern, StripSubdetector::SubDetector substructure);
    static uint16_t encode(const TrackingRecHit &hit);
    static uint16_t encode(const DetId &id, TrackingRecHit::Type hitType);
    // generic count methods
    typedef bool filterType(uint16_t);
    int countHits(HitCategory category, filterType filter) const;
    int countTypedHits(HitCategory category, filterType typeFilter, filterType filter) const;

    bool insertTrackHit(const uint16_t pattern);
    bool insertExpectedInnerHit(const uint16_t pattern);
    bool insertExpectedOuterHit(const uint16_t pattern);

    uint16_t getHitPatternByAbsoluteIndex(int position) const;

    std::pair<uint8_t, uint8_t> getCategoryIndexRange(HitCategory category) const;

    uint16_t hitPattern[MaxHits];
    uint8_t hitCount;

    uint8_t beginTrackHits;
    uint8_t endTrackHits;
    uint8_t beginInner;
    uint8_t endInner;
    uint8_t beginOuter;
    uint8_t endOuter;
};

inline std::pair<uint8_t, uint8_t> HitPattern::getCategoryIndexRange(HitCategory category) const
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

template<typename I>
bool HitPattern::appendHits(const I &begin, const I &end)
{
    for (I hit = begin; hit != end; hit++) {
        if unlikely((!appendHit(*hit))) {
            return false;
        }
    }
    return true;
}

inline bool HitPattern::pixelHitFilter(uint16_t pattern)
{
    if unlikely(!trackerHitFilter(pattern)) {
        return false;
    }

    uint32_t substructure = getSubStructure(pattern);
    return (substructure == PixelSubdetector::PixelBarrel ||
            substructure == PixelSubdetector::PixelEndcap);
}

inline bool HitPattern::pixelBarrelHitFilter(uint16_t pattern)
{
    if unlikely(!trackerHitFilter(pattern)) {
        return false;
    }

    uint32_t substructure = getSubStructure(pattern);
    return (substructure == PixelSubdetector::PixelBarrel);
}

inline bool HitPattern::pixelEndcapHitFilter(uint16_t pattern)
{
    if unlikely(!trackerHitFilter(pattern)) {
        return false;
    }

    uint32_t substructure = getSubStructure(pattern);
    return (substructure == PixelSubdetector::PixelEndcap);
}

inline bool HitPattern::stripHitFilter(uint16_t pattern)
{
    if unlikely(!trackerHitFilter(pattern)) {
        return false;
    }

    uint32_t substructure = getSubStructure(pattern);
    return (substructure == StripSubdetector::TIB ||
            substructure == StripSubdetector::TID ||
            substructure == StripSubdetector::TOB ||
            substructure == StripSubdetector::TEC);
}

inline bool HitPattern::stripSubdetectorHitFilter(uint16_t pattern, StripSubdetector::SubDetector substructure)
{
    if unlikely(!trackerHitFilter(pattern)) {
        return false;
    }

    return substructure == getSubStructure(pattern);
}

inline bool HitPattern::stripTIBHitFilter(uint16_t pattern)
{
    return stripSubdetectorHitFilter(pattern, StripSubdetector::TIB);
}

inline bool HitPattern::stripTIDHitFilter(uint16_t pattern)
{
    return stripSubdetectorHitFilter(pattern, StripSubdetector::TID);
}

inline bool HitPattern::stripTOBHitFilter(uint16_t pattern)
{
    return stripSubdetectorHitFilter(pattern, StripSubdetector::TOB);
}

inline bool HitPattern::stripTECHitFilter(uint16_t pattern)
{
    return stripSubdetectorHitFilter(pattern, StripSubdetector::TEC);
}

inline bool HitPattern::muonDTHitFilter(uint16_t pattern)
{
    if unlikely(!muonHitFilter(pattern)) {
        return false;
    }

    uint32_t substructure = getSubStructure(pattern);
    return (substructure == (uint32_t) MuonSubdetId::DT);
}

inline bool HitPattern::muonCSCHitFilter(uint16_t pattern)
{
    if unlikely(!muonHitFilter(pattern)) {
        return false;
    }

    uint32_t substructure = getSubStructure(pattern);
    return (substructure == (uint32_t) MuonSubdetId::CSC);
}

inline bool HitPattern::muonRPCHitFilter(uint16_t pattern)
{
    if unlikely(!muonHitFilter(pattern)) {
        return false;
    }

    uint32_t substructure = getSubStructure(pattern);
    return (substructure == (uint32_t) MuonSubdetId::RPC);
}

inline bool HitPattern::trackerHitFilter(uint16_t pattern)
{
    if unlikely(pattern == HitPattern::EMPTY_PATTERN) {
        return false;
    }

    return (((pattern >> SubDetectorOffset) & SubDetectorMask) == 1);
}

inline bool HitPattern::muonHitFilter(uint16_t pattern)
{
    if unlikely(pattern == HitPattern::EMPTY_PATTERN) {
        return false;
    }

    return (((pattern >> SubDetectorOffset) & SubDetectorMask) == 0);
}

inline uint32_t HitPattern::getSubStructure(uint16_t pattern)
{
    if unlikely(pattern == HitPattern::EMPTY_PATTERN) {
        return NULL_RETURN;
    }

    return ((pattern >> SubstrOffset) & SubstrMask);
}

inline uint32_t HitPattern::getLayer(uint16_t pattern)
{
    return HitPattern::getSubSubStructure(pattern);
}

inline uint32_t HitPattern::getSubSubStructure(uint16_t pattern)
{
    if unlikely(pattern == HitPattern::EMPTY_PATTERN) {
        return NULL_RETURN;
    }

    return ((pattern >> LayerOffset) & LayerMask);
}

inline uint32_t HitPattern::getSubDetector(uint16_t pattern)
{
    if unlikely(pattern == HitPattern::EMPTY_PATTERN) {
        return NULL_RETURN;
    }

    return ((pattern >> SubDetectorOffset) & SubDetectorMask);
}


inline uint32_t HitPattern::getSide(uint16_t pattern)
{
    if unlikely(pattern == HitPattern::EMPTY_PATTERN) {
        return NULL_RETURN;
    }

    return (pattern >> SideOffset) & SideMask;
}

inline uint32_t HitPattern::getHitType(uint16_t pattern)
{
    if unlikely(pattern == HitPattern::EMPTY_PATTERN) {
        return NULL_RETURN;
    }

    return ((pattern >> HitTypeOffset) & HitTypeMask);
}

inline uint16_t HitPattern::getMuonStation(uint16_t pattern)
{
    return (getSubSubStructure(pattern) >> 2) + 1;
}

inline uint16_t HitPattern::getDTSuperLayer(uint16_t pattern)
{
    return (getSubSubStructure(pattern) & 3);
}

inline uint16_t HitPattern::getCSCRing(uint16_t pattern)
{
    return (getSubSubStructure(pattern) & 3) + 1;
}

inline uint16_t HitPattern::getRPCLayer(uint16_t pattern)
{
    uint16_t subSubStructure = getSubSubStructure(pattern);
    uint16_t stat = subSubStructure >> 2;

    if likely(stat <= 1) {
        return ((subSubStructure >> 1) & 1) + 1;
    }

    return 0;
}

inline uint16_t HitPattern::getRPCregion(uint16_t pattern)
{
    return getSubSubStructure(pattern) & 1;
}

inline bool HitPattern::validHitFilter(uint16_t pattern)
{
    return getHitType(pattern) == HitPattern::VALID;
}

inline bool HitPattern::missingHitFilter(uint16_t pattern)
{
    uint16_t hitType = getHitType(pattern);
    return (hitType == HitPattern::MISSING ||
            hitType == HitPattern::MISSING_INNER ||
            hitType == HitPattern::MISSING_OUTER);
}

inline bool HitPattern::inactiveHitFilter(uint16_t pattern)
{
    return getHitType(pattern) == HitPattern::INACTIVE;
}

inline bool HitPattern::badHitFilter(uint16_t pattern)
{
    return getHitType(pattern) == HitPattern::BAD;
}

inline bool HitPattern::expectedInnerHitFilter(uint16_t pattern)
{
    return getHitType(pattern) == HitPattern::MISSING_INNER;
}

inline bool HitPattern::expectedOuterHitFilter(uint16_t pattern)
{
    return getHitType(pattern) == HitPattern::MISSING_OUTER;
}

inline int HitPattern::numberOfTrackerHits(HitCategory category) const
{
    return countHits(category, trackerHitFilter);
}

inline int HitPattern::numberOfMuonHits() const
{
    return countHits(TRACK_HITS, muonHitFilter);
}

inline int HitPattern::numberOfValidHits() const
{
    return countHits(TRACK_HITS, validHitFilter);
}

inline int HitPattern::numberOfValidTrackerHits() const
{
    return countTypedHits(TRACK_HITS, validHitFilter, trackerHitFilter);
}

inline int HitPattern::numberOfValidMuonHits() const
{
    return countTypedHits(TRACK_HITS, validHitFilter, muonHitFilter);
}

inline int HitPattern::numberOfValidPixelHits() const
{
    return countTypedHits(TRACK_HITS, validHitFilter, pixelHitFilter);
}

inline int HitPattern::numberOfValidPixelBarrelHits() const
{
    return countTypedHits(TRACK_HITS, validHitFilter, pixelBarrelHitFilter);
}

inline int HitPattern::numberOfValidPixelEndcapHits() const
{
    return countTypedHits(TRACK_HITS, validHitFilter, pixelEndcapHitFilter);
}

inline int HitPattern::numberOfValidStripHits() const
{
    return countTypedHits(TRACK_HITS, validHitFilter, stripHitFilter);
}

inline int HitPattern::numberOfValidStripTIBHits() const
{
    return countTypedHits(TRACK_HITS, validHitFilter, stripTIBHitFilter);
}

inline int HitPattern::numberOfValidStripTIDHits() const
{
    return countTypedHits(TRACK_HITS, validHitFilter, stripTIDHitFilter);
}

inline int HitPattern::numberOfValidStripTOBHits() const
{
    return countTypedHits(TRACK_HITS, validHitFilter, stripTOBHitFilter);
}

inline int HitPattern::numberOfValidStripTECHits() const
{
    return countTypedHits(TRACK_HITS, validHitFilter, stripTECHitFilter);
}

inline int HitPattern::numberOfValidMuonDTHits() const
{
    return countTypedHits(TRACK_HITS, validHitFilter, muonDTHitFilter);
}

inline int HitPattern::numberOfValidMuonCSCHits() const
{
    return countTypedHits(TRACK_HITS, validHitFilter, muonCSCHitFilter);
}

inline int HitPattern::numberOfValidMuonRPCHits() const
{
    return countTypedHits(TRACK_HITS, validHitFilter, muonRPCHitFilter);
}

inline int HitPattern::numberOfLostHits(HitCategory category) const
{
    return countHits(category, missingHitFilter);
}

inline int HitPattern::numberOfLostTrackerHits(HitCategory category) const
{
    return countTypedHits(category, missingHitFilter, trackerHitFilter);
}

inline int HitPattern::numberOfLostMuonHits() const
{
    return countTypedHits(TRACK_HITS, missingHitFilter, muonHitFilter);
}

inline int HitPattern::numberOfLostPixelHits(HitCategory category) const
{
    return countTypedHits(category, missingHitFilter, pixelHitFilter);
}

inline int HitPattern::numberOfLostPixelBarrelHits(HitCategory category) const
{
    return countTypedHits(category, missingHitFilter, pixelBarrelHitFilter);
}

inline int HitPattern::numberOfLostPixelEndcapHits(HitCategory category) const
{
    return countTypedHits(category, missingHitFilter, pixelEndcapHitFilter);
}

inline int HitPattern::numberOfLostStripHits(HitCategory category) const
{
    return countTypedHits(category, missingHitFilter, stripHitFilter);
}

inline int HitPattern::numberOfLostStripTIBHits(HitCategory category) const
{
    return countTypedHits(category, missingHitFilter, stripTIBHitFilter);
}

inline int HitPattern::numberOfLostStripTIDHits(HitCategory category) const
{
    return countTypedHits(category, missingHitFilter, stripTIDHitFilter);
}

inline int HitPattern::numberOfLostStripTOBHits(HitCategory category) const
{
    return countTypedHits(category, missingHitFilter, stripTOBHitFilter);
}

inline int HitPattern::numberOfLostStripTECHits(HitCategory category) const
{
    return countTypedHits(category, missingHitFilter, stripTECHitFilter);
}

inline int HitPattern::numberOfLostMuonDTHits() const
{
    return countTypedHits(TRACK_HITS, missingHitFilter, muonDTHitFilter);
}

inline int HitPattern::numberOfLostMuonCSCHits() const
{
    return countTypedHits(TRACK_HITS, missingHitFilter, muonCSCHitFilter);
}

inline int HitPattern::numberOfLostMuonRPCHits() const
{
    return countTypedHits(TRACK_HITS, missingHitFilter, muonRPCHitFilter);
}

inline int HitPattern::numberOfBadHits() const
{
    return countHits(TRACK_HITS, badHitFilter);
}

inline int HitPattern::numberOfBadMuonHits() const
{
    return countTypedHits(TRACK_HITS, inactiveHitFilter, muonHitFilter);
}

inline int HitPattern::numberOfBadMuonDTHits() const
{
    return countTypedHits(TRACK_HITS, inactiveHitFilter, muonDTHitFilter);
}

inline int HitPattern::numberOfBadMuonCSCHits() const
{
    return countTypedHits(TRACK_HITS, inactiveHitFilter, muonCSCHitFilter);
}

inline int HitPattern::numberOfBadMuonRPCHits() const
{
    return countTypedHits(TRACK_HITS, inactiveHitFilter, muonRPCHitFilter);
}

inline int HitPattern::numberOfInactiveHits(HitCategory category) const
{
    return countHits(category, inactiveHitFilter);
}

inline int HitPattern::numberOfInactiveTrackerHits(HitCategory category) const
{
    return countTypedHits(category, inactiveHitFilter, trackerHitFilter);
}

inline int HitPattern::numberOfExpectedInnerHits(HitCategory category) const
{
    return countTypedHits(category, expectedInnerHitFilter, trackerHitFilter);
}

inline int HitPattern::numberOfExpectedOuterHits(HitCategory category) const
{
    return countTypedHits(category, expectedOuterHitFilter, trackerHitFilter);
}

inline int HitPattern::trackerLayersWithMeasurement(HitCategory category) const
{
    return pixelLayersWithMeasurement(category) +
           stripLayersWithMeasurement(category);
}

inline int HitPattern::pixelLayersWithMeasurement(HitCategory category) const
{
    return pixelBarrelLayersWithMeasurement(category) +
           pixelEndcapLayersWithMeasurement(category);
}

inline int HitPattern::stripLayersWithMeasurement(HitCategory category) const
{
    return stripTIBLayersWithMeasurement(category) +
           stripTIDLayersWithMeasurement(category) +
           stripTOBLayersWithMeasurement(category) +
           stripTECLayersWithMeasurement(category);
}

inline int HitPattern::trackerLayersWithoutMeasurement(HitCategory category) const
{
    return pixelLayersWithoutMeasurement(category) +
           stripLayersWithoutMeasurement(category);
}

inline int HitPattern::pixelLayersWithoutMeasurement(HitCategory category) const
{
    return pixelBarrelLayersWithoutMeasurement(category) +
           pixelEndcapLayersWithoutMeasurement(category);
}

inline int HitPattern::stripLayersWithoutMeasurement(HitCategory category) const
{
    return stripTIBLayersWithoutMeasurement(category) +
           stripTIDLayersWithoutMeasurement(category) +
           stripTOBLayersWithoutMeasurement(category) +
           stripTECLayersWithoutMeasurement(category);
}

inline int HitPattern::trackerLayersTotallyOffOrBad(HitCategory category) const
{
    return pixelLayersTotallyOffOrBad(category) +
           stripLayersTotallyOffOrBad(category);
}

inline int HitPattern::pixelLayersTotallyOffOrBad(HitCategory category) const
{
    return pixelBarrelLayersTotallyOffOrBad(category) +
           pixelEndcapLayersTotallyOffOrBad(category);
}

inline int HitPattern::stripLayersTotallyOffOrBad(HitCategory category) const
{
    return stripTIBLayersTotallyOffOrBad(category) +
           stripTIDLayersTotallyOffOrBad(category) +
           stripTOBLayersTotallyOffOrBad(category) +
           stripTECLayersTotallyOffOrBad(category);
}

inline int HitPattern::trackerLayersNull(HitCategory category) const
{
    return pixelLayersNull(category) +
           stripLayersNull(category);
}

inline int HitPattern::pixelLayersNull(HitCategory category) const
{
    return pixelBarrelLayersNull(category) +
           pixelEndcapLayersNull(category);
}

inline int HitPattern::stripLayersNull(HitCategory category) const
{
    return stripTIBLayersNull(category) +
           stripTIDLayersNull(category) +
           stripTOBLayersNull(category) +
           stripTECLayersNull(category);
}

inline int HitPattern::muonStationsWithValidHits() const
{
    return muonStations(0, 0);
}

inline int HitPattern::muonStationsWithBadHits() const
{
    return muonStations(0, 3);
}

inline int HitPattern::muonStationsWithAnyHits() const
{
    return muonStations(0, -1);
}

inline int HitPattern::dtStationsWithValidHits() const
{
    return muonStations(1, 0);
}

inline int HitPattern::dtStationsWithBadHits() const
{
    return muonStations(1, 3);
}

inline int HitPattern::dtStationsWithAnyHits() const
{
    return muonStations(1, -1);
}

inline int HitPattern::cscStationsWithValidHits() const
{
    return muonStations(2, 0);
}

inline int HitPattern::cscStationsWithBadHits() const
{
    return muonStations(2, 3);
}

inline int HitPattern::cscStationsWithAnyHits() const
{
    return muonStations(2, -1);
}

inline int HitPattern::rpcStationsWithValidHits() const
{
    return muonStations(3, 0);
}

inline int HitPattern::rpcStationsWithBadHits() const
{
    return muonStations(3, 3);
}

inline int HitPattern::rpcStationsWithAnyHits() const
{
    return muonStations(3, -1);
}

inline int HitPattern::innermostMuonStationWithValidHits() const
{
    return innermostMuonStationWithHits(0);
}

inline int HitPattern::innermostMuonStationWithBadHits() const
{
    return innermostMuonStationWithHits(3);
}

inline int HitPattern::innermostMuonStationWithAnyHits() const
{
    return innermostMuonStationWithHits(-1);
}

inline int HitPattern::outermostMuonStationWithValidHits() const
{
    return outermostMuonStationWithHits(0);
}

inline int HitPattern::outermostMuonStationWithBadHits() const
{
    return outermostMuonStationWithHits(3);
}

inline int HitPattern::outermostMuonStationWithAnyHits() const
{
    return outermostMuonStationWithHits(-1);
}

} // namespace reco

#endif

