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
// +------------+---------------+---------------------------+-----------------+----------------+
// |  tk/mu/mtd | sub-structure |     sub-sub-structure     |     stereo      |    hit type    |
// +------------+---------------+---------------------------+-----------------+----------------+
// |    11-10   | 9   8    7    |  6     5     4     3      |        2        |    1        0  |  bit
// +------------+---------------+---------------------------+-----------------+----------------|
// | tk  = 1    |    PXB = 1    | layer = 1-3               |                 | hit type = 0-3 |
// | tk  = 1    |    PXF = 2    | disk  = 1-2               |                 | hit type = 0-3 |
// | tk  = 1    |    TIB = 3    | layer = 1-4               | 0=rphi,1=stereo | hit type = 0-3 |
// | tk  = 1    |    TID = 4    | wheel = 1-3               | 0=rphi,1=stereo | hit type = 0-3 |
// | tk  = 1    |    TOB = 5    | layer = 1-6               | 0=rphi,1=stereo | hit type = 0-3 |
// | tk  = 1    |    TEC = 6    | wheel = 1-9               | 0=rphi,1=stereo | hit type = 0-3 |
// | mu  = 0    |    DT  = 1    | 4*(stat-1)+superlayer     |                 | hit type = 0-3 |
// | mu  = 0    |    CSC = 2    | 4*(stat-1)+(ring-1)       |                 | hit type = 0-3 |
// | mu  = 0    |    RPC = 3    | 4*(stat-1)+2*layer+region |                 | hit type = 0-3 |
// | mu  = 0    |    GEM = 4    | 2*(stat-1)+2*(layer-1)    |                 | hit type = 0-3 |
// | mu  = 0    |    ME0 = 5    | roll                      |                 | hit type = 0-3 |
// | mtd = 2    |    BTL = 1    | moduleType = 1-3          |                 | hit type = 0-3 |
// | mtd = 2    |    ETL = 2    | ring = 1-12               |                 | hit type = 0-3 |
// +------------+---------------+---------------------------+-----------------+----------------+
//
//  hit type, see DataFormats/TrackingRecHit/interface/TrackingRecHit.h
//      VALID    = valid hit                                     = 0
//      MISSING  = detector is good, but no rec hit found        = 1
//      INACTIVE = detector is off, so there was no hope         = 2
//      BAD      = there were many bad strips within the ellipse = 3
//
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
//        layer case 0: VALID + (MISSING, OFF, BAD) ==> with measurement
//        layer case 1: MISSING + (OFF, BAD) ==> without measurement
//        layer case 2: OFF, BAD ==> totally off or bad, cannot say much
//     not crossed
//        layer case NULL_RETURN: track outside acceptance or in gap ==> null
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
//
// Given a track, here is an example usage of hit pattern
//
//     // hit pattern of the track
//    const reco::HitPattern &p = track->hitPattern();
//
//     // loop over the hits of the track.
//    for (int i = 0; i < p.numberOfAllHits(HitPattern::TRACK_HITS); i++) {
//        uint32_t hit = p.getHitPattern(HitPattern::TRACK_HITS, i);
//
//        // if the hit is valid and in pixel barrel, print out the layer
//        if (p.validHitFilter(hit) && p.pixelBarrelHitFilter(hit)){
//            cout << "valid hit found in pixel barrel layer "
//                 << p.getLayer(hit)
//                 << endl;
//        }
//
//        // expert level: printout the hit in 11-bit binary format
//        cout << "hit in 11-bit binary format = ";
//        for (int j = 10; j >= 0; j--){
//            int bit = (hit >> j) & 0x1;
//            cout << bit;
//        }
//        cout << endl;
//    }
//
//    //count the number of valid pixel barrel *** hits ***
//    cout << "number of of valid pixel barrel hits is "
//         << p.numberOfValidPixelBarrelHits()
//         << endl;
//
//    //count the number of pixel barrel *** layers *** with measurement
//    cout << "number of of pixel barrel layers with measurement is "
//         << p.pixelBarrelLayersWithMeasurement()
//         << endl;
//

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/ForwardDetId/interface/MTDDetId.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "FWCore/Utilities/interface/Likely.h"

#include <utility>
#include <algorithm>
#include <iostream>
#include <ostream>
#include <memory>

class TrackerTopology;

namespace test {
  namespace TestHitPattern {
    int test();
  }
}  // namespace test

namespace reco {

  class HitPattern {
  public:
    enum { MONO = 1, STEREO = 2 };

    enum HIT_DETECTOR_TYPE { MUON_HIT = 0, TRACKER_HIT = 1, MTD_HIT = 2 };

    enum HIT_TYPE { VALID = 0, MISSING = 1, INACTIVE = 2, BAD = 3 };

    enum HitCategory { TRACK_HITS = 0, MISSING_INNER_HITS = 1, MISSING_OUTER_HITS = 2 };
    const static unsigned short ARRAY_LENGTH = 57;
    const static unsigned short HIT_LENGTH = 12;
    const static unsigned short MaxHits = (8 * sizeof(uint16_t) * ARRAY_LENGTH) / HIT_LENGTH;

    static const uint32_t NULL_RETURN = 999999;
    static const uint16_t EMPTY_PATTERN = 0x0;

    static bool trackerHitFilter(uint16_t pattern);
    static bool muonHitFilter(uint16_t pattern);
    static bool timingHitFilter(uint16_t pattern);

    static bool validHitFilter(uint16_t pattern);
    static bool missingHitFilter(uint16_t pattern);
    static bool inactiveHitFilter(uint16_t pattern);
    static bool badHitFilter(uint16_t pattern);

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
    static bool muonGEMHitFilter(uint16_t pattern);
    static bool muonME0HitFilter(uint16_t pattern);

    static bool timingBTLHitFilter(uint16_t pattern);
    static bool timingETLHitFilter(uint16_t pattern);

    static uint32_t getHitType(uint16_t pattern);

    // mono (0) or stereo (1)
    static uint32_t getSide(uint16_t pattern);
    static uint32_t getLayer(uint16_t pattern);
    static uint32_t getSubSubStructure(uint16_t pattern);
    static uint32_t getSubStructure(uint16_t pattern);
    static uint32_t getSubDetector(uint16_t pattern);

    /// Muon station (1-4). Only valid for muon patterns, of course. only for patterns from muon, of course
    static uint16_t getMuonStation(uint16_t pattern);

    /// DT superlayer (1-3). Where the "hit" was a DT segment, superlayer is 0. Only valid for muon DT patterns, of course.
    static uint16_t getDTSuperLayer(uint16_t pattern);  // only for DT patterns

    /// CSC ring (1-4). Only valid for muon CSC patterns, of course.
    static uint16_t getCSCRing(uint16_t pattern);

    /// RPC layer: for station 1 and 2, layer = 1(inner) or 2(outer); for station 3, 4 layer is always 0. Only valid for muon RPC patterns, of course.
    static uint16_t getRPCLayer(uint16_t pattern);

    /// RPC region: 0 = barrel, 1 = endcap. Only valid for muon RPC patterns, of course.
    static uint16_t getRPCregion(uint16_t pattern);

    /// GEM station: 1,2. Only valid for muon GEM patterns, of course.
    static uint16_t getGEMStation(uint16_t pattern);

    /// GEM layer: 1,2. Only valid for muon GEM patterns, of course.
    static uint16_t getGEMLayer(uint16_t pattern);

    /// BTL Module type: 1,2,3. Only valid for BTL patterns of course.
    static uint16_t getBTLModType(uint16_t pattern);

    /// ETL Ring: 1-12. Only valid for ETL patterns of course.
    static uint16_t getETLRing(uint16_t pattern);

    HitPattern();

    ~HitPattern();

    HitPattern(const HitPattern &other);

    HitPattern &operator=(const HitPattern &other);

    template <typename I>
    bool appendHits(const I &begin, const I &end, const TrackerTopology &ttopo);
    bool appendHit(const TrackingRecHit &hit, const TrackerTopology &ttopo);
    bool appendHit(const TrackingRecHitRef &ref, const TrackerTopology &ttopo);
    bool appendHit(const DetId &id, TrackingRecHit::Type hitType, const TrackerTopology &ttopo);
    bool appendHit(const uint16_t pattern, TrackingRecHit::Type hitType);

    /**
     * This is meant to be used only in cases where the an
     * already-packed hit information is re-interpreted in terms of
     * HitPattern (i.e. MiniAOD PackedCandidate, and the IO rule for
     * reading old versions of HitPattern)
     */
    bool appendTrackerHit(uint16_t subdet, uint16_t layer, uint16_t stereo, TrackingRecHit::Type hitType);

    /**
     * This is meant to be used only in cases where the an
     * already-packed hit information is re-interpreted in terms of
     * HitPattern (i.e. the IO rule for reading old versions of
     * HitPattern)
     */
    bool appendMuonHit(const DetId &id, TrackingRecHit::Type hitType);

    // get the pattern of the position-th hit
    uint16_t getHitPattern(HitCategory category, int position) const;

    void clear();

    // print the pattern of the position-th hit
    void printHitPattern(HitCategory category, int position, std::ostream &stream) const;
    void print(HitCategory category, std::ostream &stream = std::cout) const;

    // has valid hit in PXB/PXF layer x
    bool hasValidHitInPixelLayer(enum PixelSubdetector::SubDetector, uint16_t layer) const;

    int numberOfAllHits(HitCategory category) const;  // not-null
    int numberOfValidHits() const;                    // not-null, valid

    int numberOfAllTrackerHits(HitCategory category) const;  // not-null, tracker
    int numberOfValidTrackerHits() const;                    // not-null, valid, tracker
    int numberOfValidPixelHits() const;                      // not-null, valid, pixel
    int numberOfValidPixelBarrelHits() const;                // not-null, valid, pixel PXB
    int numberOfValidPixelEndcapHits() const;                // not-null, valid, pixel PXF
    int numberOfValidStripHits() const;                      // not-null, valid, strip
    int numberOfValidStripTIBHits() const;                   // not-null, valid, strip TIB
    int numberOfValidStripTIDHits() const;                   // not-null, valid, strip TID
    int numberOfValidStripTOBHits() const;                   // not-null, valid, strip TOB
    int numberOfValidStripTECHits() const;                   // not-null, valid, strip TEC

    int numberOfLostHits(HitCategory category) const;             // not-null, not valid
    int numberOfLostTrackerHits(HitCategory category) const;      // not-null, not valid, tracker
    int numberOfLostPixelHits(HitCategory category) const;        // not-null, not valid, pixel
    int numberOfLostPixelBarrelHits(HitCategory category) const;  // not-null, not valid, pixel PXB
    int numberOfLostPixelEndcapHits(HitCategory category) const;  // not-null, not valid, pixel PXF
    int numberOfLostStripHits(HitCategory category) const;        // not-null, not valid, strip
    int numberOfLostStripTIBHits(HitCategory category) const;     // not-null, not valid, strip TIB
    int numberOfLostStripTIDHits(HitCategory category) const;     // not-null, not valid, strip TID
    int numberOfLostStripTOBHits(HitCategory category) const;     // not-null, not valid, strip TOB
    int numberOfLostStripTECHits(HitCategory category) const;     // not-null, not valid, strip TEC

    int numberOfTimingHits() const;          // not-null timing
    int numberOfValidTimingHits() const;     // not-null, valid, timing
    int numberOfValidTimingBTLHits() const;  // not-null, valid, timing BTL
    int numberOfValidTimingETLHits() const;  // not-null, valid, timing ETL

    int numberOfLostTimingHits() const;     // not-null, not valid, timing
    int numberOfLostTimingBTLHits() const;  // not-null, not valid, timing BTL
    int numberOfLostTimingETLHits() const;  // not-null, not valid, timing ETL

    int numberOfMuonHits() const;          // not-null, muon
    int numberOfValidMuonHits() const;     // not-null, valid, muon
    int numberOfValidMuonDTHits() const;   // not-null, valid, muon DT
    int numberOfValidMuonCSCHits() const;  // not-null, valid, muon CSC
    int numberOfValidMuonRPCHits() const;  // not-null, valid, muon RPC
    int numberOfValidMuonGEMHits() const;  // not-null, valid, muon GEM
    int numberOfValidMuonME0Hits() const;  // not-null, valid, muon ME0

    int numberOfLostMuonHits() const;     // not-null, not valid, muon
    int numberOfLostMuonDTHits() const;   // not-null, not valid, muon DT
    int numberOfLostMuonCSCHits() const;  // not-null, not valid, muon CSC
    int numberOfLostMuonRPCHits() const;  // not-null, not valid, muon RPC
    int numberOfLostMuonGEMHits() const;  // not-null, not valid, muon GEM
    int numberOfLostMuonME0Hits() const;  // not-null, not valid, muon ME0

    int numberOfBadHits() const;         // not-null, bad (only used in Muon Ch.)
    int numberOfBadMuonHits() const;     // not-null, bad, muon
    int numberOfBadMuonDTHits() const;   // not-null, bad, muon DT
    int numberOfBadMuonCSCHits() const;  // not-null, bad, muon CSC
    int numberOfBadMuonRPCHits() const;  // not-null, bad, muon RPC
    int numberOfBadMuonGEMHits() const;  // not-null, bad, muon GEM
    int numberOfBadMuonME0Hits() const;  // not-null, bad, muon ME0

    int numberOfInactiveHits() const;         // not-null, inactive
    int numberOfInactiveTrackerHits() const;  // not-null, inactive, tracker
    int numberOfInactiveTimingHits() const;   // not-null, inactive, timing

    // count strip layers that have non-null, valid mono and stereo hits
    int numberOfValidStripLayersWithMonoAndStereo(uint16_t stripdet, uint16_t layer) const;
    int numberOfValidStripLayersWithMonoAndStereo() const;
    int numberOfValidTOBLayersWithMonoAndStereo(uint32_t layer = 0) const;
    int numberOfValidTIBLayersWithMonoAndStereo(uint32_t layer = 0) const;
    int numberOfValidTIDLayersWithMonoAndStereo(uint32_t layer = 0) const;
    int numberOfValidTECLayersWithMonoAndStereo(uint32_t layer = 0) const;

    uint32_t getTrackerLayerCase(HitCategory category, uint16_t substr, uint16_t layer) const;
    uint16_t getTrackerMonoStereo(HitCategory category, uint16_t substr, uint16_t layer) const;

    int trackerLayersWithMeasurementOld() const;   // case 0: tracker
    int trackerLayersWithMeasurement() const;      // case 0: tracker
    int pixelLayersWithMeasurementOld() const;     // case 0: pixel
    int pixelLayersWithMeasurement() const;        // case 0: pixel
    int stripLayersWithMeasurement() const;        // case 0: strip
    int pixelBarrelLayersWithMeasurement() const;  // case 0: pixel PXB
    int pixelEndcapLayersWithMeasurement() const;  // case 0: pixel PXF
    int stripTIBLayersWithMeasurement() const;     // case 0: strip TIB
    int stripTIDLayersWithMeasurement() const;     // case 0: strip TID
    int stripTOBLayersWithMeasurement() const;     // case 0: strip TOB
    int stripTECLayersWithMeasurement() const;     // case 0: strip TEC

    int trackerLayersWithoutMeasurement(HitCategory category) const;      // case 1: tracker
    int trackerLayersWithoutMeasurementOld(HitCategory category) const;   // case 1: tracker
    int pixelLayersWithoutMeasurement(HitCategory category) const;        // case 1: pixel
    int stripLayersWithoutMeasurement(HitCategory category) const;        // case 1: strip
    int pixelBarrelLayersWithoutMeasurement(HitCategory category) const;  // case 1: pixel PXB
    int pixelEndcapLayersWithoutMeasurement(HitCategory category) const;  // case 1: pixel PXF
    int stripTIBLayersWithoutMeasurement(HitCategory category) const;     // case 1: strip TIB
    int stripTIDLayersWithoutMeasurement(HitCategory category) const;     // case 1: strip TID
    int stripTOBLayersWithoutMeasurement(HitCategory category) const;     // case 1: strip TOB
    int stripTECLayersWithoutMeasurement(HitCategory category) const;     // case 1: strip TEC

    int trackerLayersTotallyOffOrBad(HitCategory category = TRACK_HITS) const;      // case 2: tracker
    int pixelLayersTotallyOffOrBad(HitCategory category = TRACK_HITS) const;        // case 2: pixel
    int stripLayersTotallyOffOrBad(HitCategory category = TRACK_HITS) const;        // case 2: strip
    int pixelBarrelLayersTotallyOffOrBad(HitCategory category = TRACK_HITS) const;  // case 2: pixel PXB
    int pixelEndcapLayersTotallyOffOrBad(HitCategory category = TRACK_HITS) const;  // case 2: pixel PXF
    int stripTIBLayersTotallyOffOrBad(HitCategory category = TRACK_HITS) const;     // case 2: strip TIB
    int stripTIDLayersTotallyOffOrBad(HitCategory category = TRACK_HITS) const;     // case 2: strip TID
    int stripTOBLayersTotallyOffOrBad(HitCategory category = TRACK_HITS) const;     // case 2: strip TOB
    int stripTECLayersTotallyOffOrBad(HitCategory category = TRACK_HITS) const;     // case 2: strip TEC

    int trackerLayersNull() const;      // case NULL_RETURN: tracker
    int pixelLayersNull() const;        // case NULL_RETURN: pixel
    int stripLayersNull() const;        // case NULL_RETURN: strip
    int pixelBarrelLayersNull() const;  // case NULL_RETURN: pixel PXB
    int pixelEndcapLayersNull() const;  // case NULL_RETURN: pixel PXF
    int stripTIBLayersNull() const;     // case NULL_RETURN: strip TIB
    int stripTIDLayersNull() const;     // case NULL_RETURN: strip TID
    int stripTOBLayersNull() const;     // case NULL_RETURN: strip TOB
    int stripTECLayersNull() const;     // case NULL_RETURN: strip TEC

    /// subdet = 0(all), 1(DT), 2(CSC), 3(RPC) 4(GEM); hitType=-1(all), 0=valid, 3=bad
    int muonStations(int subdet, int hitType) const;

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

    int gemStationsWithValidHits() const;
    int gemStationsWithBadHits() const;
    int gemStationsWithAnyHits() const;

    int me0StationsWithValidHits() const;
    int me0StationsWithBadHits() const;
    int me0StationsWithAnyHits() const;

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

    //only used by ROOT IO rule to read v12 HitPatterns
    static bool fillNewHitPatternWithOldHitPattern_v12(const uint16_t oldHitPattern[],
                                                       uint8_t hitCount,
                                                       uint8_t beginTrackHits,
                                                       uint8_t endTrackHits,
                                                       uint8_t beginInner,
                                                       uint8_t endInner,
                                                       uint8_t beginOuter,
                                                       uint8_t endOuter,
                                                       reco::HitPattern *newObj);

  private:
    // 3 bits for hit type
    const static unsigned short HitTypeOffset = 0;
    const static unsigned short HitTypeMask = 0x3;

    // 1 bit to identify the side in double-sided detectors
    const static unsigned short SideOffset = 2;
    const static unsigned short SideMask = 0x1;

    // 4 bits to identify the layer/disk/wheel within the substructure
    const static unsigned short LayerOffset = 3;
    const static unsigned short LayerMask = 0xF;

    // 3 bits to identify the tracker/muon detector substructure
    const static unsigned short SubstrOffset = 7;
    const static unsigned short SubstrMask = 0x7;

    // 2 bits to distinguish tracker, muon, mtd subsystems
    const static unsigned short SubDetectorOffset = 10;
    const static unsigned short SubDetectorMask = 0x3;

    const static unsigned short minTrackerWord = 1 << SubDetectorOffset;
    const static unsigned short maxTrackerWord = (2 << SubDetectorOffset) - 1;
    const static unsigned short minPixelWord = minTrackerWord | (1 << SubstrOffset);
    const static unsigned short minStripWord = minTrackerWord | (3 << SubstrOffset);

    // detector side for tracker modules (mono/stereo)
    static uint16_t isStereo(DetId i, const TrackerTopology &ttopo);
    static bool stripSubdetectorHitFilter(uint16_t pattern, StripSubdetector::SubDetector substructure);

    static uint16_t encode(const TrackingRecHit &hit, const TrackerTopology &ttopo);
    static uint16_t encode(const DetId &id, TrackingRecHit::Type hitType, const TrackerTopology &ttopo);
    static uint16_t encode(uint16_t det, uint16_t subdet, uint16_t layer, uint16_t side, TrackingRecHit::Type hitType);

    // generic count methods
    typedef bool filterType(uint16_t);

    template <typename F>
    void call(HitCategory category, filterType typeFilter, F f) const;

    int countHits(HitCategory category, filterType filter) const;
    int countTypedHits(HitCategory category, filterType typeFilter, filterType filter) const;

    bool insertTrackHit(const uint16_t pattern);
    bool insertExpectedInnerHit(const uint16_t pattern);
    bool insertExpectedOuterHit(const uint16_t pattern);
    void insertHit(const uint16_t pattern);

    uint16_t getHitPatternByAbsoluteIndex(int position) const;

    std::pair<uint8_t, uint8_t> getCategoryIndexRange(HitCategory category) const;

    uint16_t hitPattern[ARRAY_LENGTH];
    uint8_t hitCount;

    uint8_t beginTrackHits;
    uint8_t endTrackHits;
    uint8_t beginInner;
    uint8_t endInner;
    uint8_t beginOuter;
    uint8_t endOuter;

    friend int ::test::TestHitPattern::test();

    template <int N>
    friend struct PatternSet;
  };

  inline std::pair<uint8_t, uint8_t> HitPattern::getCategoryIndexRange(HitCategory category) const {
    switch (category) {
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

  template <typename I>
  bool HitPattern::appendHits(const I &begin, const I &end, const TrackerTopology &ttopo) {
    for (I hit = begin; hit != end; hit++) {
      if UNLIKELY ((!appendHit(*hit, ttopo))) {
        return false;
      }
    }
    return true;
  }

  inline uint16_t HitPattern::getHitPattern(HitCategory category, int position) const {
    std::pair<uint8_t, uint8_t> range = getCategoryIndexRange(category);
    if UNLIKELY ((position < 0 || (position + range.first) >= range.second)) {
      return HitPattern::EMPTY_PATTERN;
    }

    return getHitPatternByAbsoluteIndex(range.first + position);
  }

  inline int HitPattern::countHits(HitCategory category, filterType filter) const {
    int count = 0;
    std::pair<uint8_t, uint8_t> range = getCategoryIndexRange(category);
    for (int i = range.first; i < range.second; ++i) {
      if (filter(getHitPatternByAbsoluteIndex(i))) {
        ++count;
      }
    }
    return count;
  }

  template <typename F>
  void HitPattern::call(HitCategory category, filterType typeFilter, F f) const {
    std::pair<uint8_t, uint8_t> range = getCategoryIndexRange(category);
    for (int i = range.first; i < range.second; i++) {
      uint16_t pattern = getHitPatternByAbsoluteIndex(i);
      // f() return false to ask to stop looping
      if (typeFilter(pattern) && !f(pattern)) {
        break;
      }
    }
  }

  inline int HitPattern::countTypedHits(HitCategory category, filterType typeFilter, filterType filter) const {
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

  inline bool HitPattern::pixelHitFilter(uint16_t pattern) {
    if UNLIKELY (!trackerHitFilter(pattern)) {
      return false;
    }

    uint32_t substructure = getSubStructure(pattern);
    return (substructure == PixelSubdetector::PixelBarrel || substructure == PixelSubdetector::PixelEndcap);
  }

  inline bool HitPattern::pixelBarrelHitFilter(uint16_t pattern) {
    if UNLIKELY (!trackerHitFilter(pattern)) {
      return false;
    }

    uint32_t substructure = getSubStructure(pattern);
    return (substructure == PixelSubdetector::PixelBarrel);
  }

  inline bool HitPattern::pixelEndcapHitFilter(uint16_t pattern) {
    if UNLIKELY (!trackerHitFilter(pattern)) {
      return false;
    }

    uint32_t substructure = getSubStructure(pattern);
    return (substructure == PixelSubdetector::PixelEndcap);
  }

  inline bool HitPattern::stripHitFilter(uint16_t pattern) {
    return pattern > minStripWord && pattern <= maxTrackerWord;
  }

  inline bool HitPattern::stripSubdetectorHitFilter(uint16_t pattern, StripSubdetector::SubDetector substructure) {
    if UNLIKELY (!trackerHitFilter(pattern)) {
      return false;
    }

    return substructure == getSubStructure(pattern);
  }

  inline bool HitPattern::stripTIBHitFilter(uint16_t pattern) {
    return stripSubdetectorHitFilter(pattern, StripSubdetector::TIB);
  }

  inline bool HitPattern::stripTIDHitFilter(uint16_t pattern) {
    return stripSubdetectorHitFilter(pattern, StripSubdetector::TID);
  }

  inline bool HitPattern::stripTOBHitFilter(uint16_t pattern) {
    return stripSubdetectorHitFilter(pattern, StripSubdetector::TOB);
  }

  inline bool HitPattern::stripTECHitFilter(uint16_t pattern) {
    return stripSubdetectorHitFilter(pattern, StripSubdetector::TEC);
  }

  inline bool HitPattern::muonDTHitFilter(uint16_t pattern) {
    if UNLIKELY (!muonHitFilter(pattern)) {
      return false;
    }

    uint32_t substructure = getSubStructure(pattern);
    return (substructure == (uint32_t)MuonSubdetId::DT);
  }

  inline bool HitPattern::muonCSCHitFilter(uint16_t pattern) {
    if UNLIKELY (!muonHitFilter(pattern)) {
      return false;
    }

    uint32_t substructure = getSubStructure(pattern);
    return (substructure == (uint32_t)MuonSubdetId::CSC);
  }

  inline bool HitPattern::muonRPCHitFilter(uint16_t pattern) {
    if UNLIKELY (!muonHitFilter(pattern)) {
      return false;
    }

    uint32_t substructure = getSubStructure(pattern);
    return (substructure == (uint32_t)MuonSubdetId::RPC);
  }

  inline bool HitPattern::muonGEMHitFilter(uint16_t pattern) {
    if UNLIKELY (!muonHitFilter(pattern)) {
      return false;
    }

    uint32_t substructure = getSubStructure(pattern);
    return (substructure == (uint32_t)MuonSubdetId::GEM);
  }

  inline bool HitPattern::muonME0HitFilter(uint16_t pattern) {
    if UNLIKELY (!muonHitFilter(pattern))
      return false;
    uint16_t substructure = getSubStructure(pattern);
    return (substructure == (uint16_t)MuonSubdetId::ME0);
  }

  inline bool HitPattern::trackerHitFilter(uint16_t pattern) {
    return pattern > minTrackerWord && pattern <= maxTrackerWord;
  }

  inline bool HitPattern::muonHitFilter(uint16_t pattern) {
    if UNLIKELY (pattern == HitPattern::EMPTY_PATTERN) {
      return false;
    }

    return (((pattern >> SubDetectorOffset) & SubDetectorMask) == 0);
  }

  inline bool HitPattern::timingBTLHitFilter(uint16_t pattern) {
    if UNLIKELY (!timingHitFilter(pattern))
      return false;
    uint16_t substructure = getSubStructure(pattern);
    return (substructure == (uint16_t)MTDDetId::BTL);
  }

  inline bool HitPattern::timingETLHitFilter(uint16_t pattern) {
    if UNLIKELY (!timingHitFilter(pattern))
      return false;
    uint16_t substructure = getSubStructure(pattern);
    return (substructure == (uint16_t)MTDDetId::ETL);
  }

  inline bool HitPattern::timingHitFilter(uint16_t pattern) {
    if UNLIKELY (pattern == HitPattern::EMPTY_PATTERN) {
      return false;
    }

    return (((pattern >> SubDetectorOffset) & SubDetectorMask) == 2);
  }

  inline uint32_t HitPattern::getSubStructure(uint16_t pattern) {
    if UNLIKELY (pattern == HitPattern::EMPTY_PATTERN) {
      return NULL_RETURN;
    }

    return ((pattern >> SubstrOffset) & SubstrMask);
  }

  inline uint32_t HitPattern::getLayer(uint16_t pattern) { return HitPattern::getSubSubStructure(pattern); }

  inline uint32_t HitPattern::getSubSubStructure(uint16_t pattern) {
    if UNLIKELY (pattern == HitPattern::EMPTY_PATTERN) {
      return NULL_RETURN;
    }

    return ((pattern >> LayerOffset) & LayerMask);
  }

  inline uint32_t HitPattern::getSubDetector(uint16_t pattern) {
    if UNLIKELY (pattern == HitPattern::EMPTY_PATTERN) {
      return NULL_RETURN;
    }

    return ((pattern >> SubDetectorOffset) & SubDetectorMask);
  }

  inline uint32_t HitPattern::getSide(uint16_t pattern) {
    if UNLIKELY (pattern == HitPattern::EMPTY_PATTERN) {
      return NULL_RETURN;
    }

    return (pattern >> SideOffset) & SideMask;
  }

  inline uint32_t HitPattern::getHitType(uint16_t pattern) {
    if UNLIKELY (pattern == HitPattern::EMPTY_PATTERN) {
      return NULL_RETURN;
    }

    return ((pattern >> HitTypeOffset) & HitTypeMask);
  }

  inline uint16_t HitPattern::getMuonStation(uint16_t pattern) { return (getSubSubStructure(pattern) >> 2) + 1; }

  inline uint16_t HitPattern::getDTSuperLayer(uint16_t pattern) { return (getSubSubStructure(pattern) & 3); }

  inline uint16_t HitPattern::getCSCRing(uint16_t pattern) { return (getSubSubStructure(pattern) & 3) + 1; }

  inline uint16_t HitPattern::getRPCLayer(uint16_t pattern) {
    uint16_t subSubStructure = getSubSubStructure(pattern);
    uint16_t stat = subSubStructure >> 2;

    if LIKELY (stat <= 1) {
      return ((subSubStructure >> 1) & 1) + 1;
    }

    return 0;
  }

  inline uint16_t HitPattern::getRPCregion(uint16_t pattern) { return getSubSubStructure(pattern) & 1; }

  ////////////////////////////// GEM
  inline uint16_t HitPattern::getGEMStation(uint16_t pattern)

  {
    uint16_t sss = getSubSubStructure(pattern), stat = sss >> 1;
    return stat + 1;
  }

  /// MTD
  inline uint16_t HitPattern::getBTLModType(uint16_t pattern) { return getSubSubStructure(pattern); }

  inline uint16_t HitPattern::getETLRing(uint16_t pattern) { return getSubSubStructure(pattern); }

  inline uint16_t HitPattern::getGEMLayer(uint16_t pattern) { return (getSubSubStructure(pattern) & 1) + 1; }

  inline bool HitPattern::validHitFilter(uint16_t pattern) { return getHitType(pattern) == HitPattern::VALID; }

  inline bool HitPattern::missingHitFilter(uint16_t pattern) { return getHitType(pattern) == HitPattern::MISSING; }

  inline bool HitPattern::inactiveHitFilter(uint16_t pattern) { return getHitType(pattern) == HitPattern::INACTIVE; }

  inline bool HitPattern::badHitFilter(uint16_t pattern) { return getHitType(pattern) == HitPattern::BAD; }

  inline int HitPattern::numberOfAllHits(HitCategory category) const {
    std::pair<uint8_t, uint8_t> range = getCategoryIndexRange(category);
    return range.second - range.first;
  }

  inline int HitPattern::numberOfAllTrackerHits(HitCategory category) const {
    return countHits(category, trackerHitFilter);
  }

  inline int HitPattern::numberOfMuonHits() const { return countHits(TRACK_HITS, muonHitFilter); }

  inline int HitPattern::numberOfTimingHits() const { return countHits(TRACK_HITS, timingHitFilter); }

  inline int HitPattern::numberOfValidHits() const { return countHits(TRACK_HITS, validHitFilter); }

  inline int HitPattern::numberOfValidTrackerHits() const {
    return countTypedHits(TRACK_HITS, validHitFilter, trackerHitFilter);
  }

  inline int HitPattern::numberOfValidMuonHits() const {
    return countTypedHits(TRACK_HITS, validHitFilter, muonHitFilter);
  }

  inline int HitPattern::numberOfValidTimingHits() const {
    return countTypedHits(TRACK_HITS, validHitFilter, timingHitFilter);
  }

  inline int HitPattern::numberOfValidPixelHits() const {
    return countTypedHits(TRACK_HITS, validHitFilter, pixelHitFilter);
  }

  inline int HitPattern::numberOfValidPixelBarrelHits() const {
    return countTypedHits(TRACK_HITS, validHitFilter, pixelBarrelHitFilter);
  }

  inline int HitPattern::numberOfValidPixelEndcapHits() const {
    return countTypedHits(TRACK_HITS, validHitFilter, pixelEndcapHitFilter);
  }

  inline int HitPattern::numberOfValidStripHits() const {
    return countTypedHits(TRACK_HITS, validHitFilter, stripHitFilter);
  }

  inline int HitPattern::numberOfValidStripTIBHits() const {
    return countTypedHits(TRACK_HITS, validHitFilter, stripTIBHitFilter);
  }

  inline int HitPattern::numberOfValidStripTIDHits() const {
    return countTypedHits(TRACK_HITS, validHitFilter, stripTIDHitFilter);
  }

  inline int HitPattern::numberOfValidStripTOBHits() const {
    return countTypedHits(TRACK_HITS, validHitFilter, stripTOBHitFilter);
  }

  inline int HitPattern::numberOfValidStripTECHits() const {
    return countTypedHits(TRACK_HITS, validHitFilter, stripTECHitFilter);
  }

  inline int HitPattern::numberOfValidMuonDTHits() const {
    return countTypedHits(TRACK_HITS, validHitFilter, muonDTHitFilter);
  }

  inline int HitPattern::numberOfValidMuonCSCHits() const {
    return countTypedHits(TRACK_HITS, validHitFilter, muonCSCHitFilter);
  }

  inline int HitPattern::numberOfValidMuonRPCHits() const {
    return countTypedHits(TRACK_HITS, validHitFilter, muonRPCHitFilter);
  }

  inline int HitPattern::numberOfValidMuonGEMHits() const {
    return countTypedHits(TRACK_HITS, validHitFilter, muonGEMHitFilter);
  }

  inline int HitPattern::numberOfValidMuonME0Hits() const {
    return countTypedHits(TRACK_HITS, validHitFilter, muonME0HitFilter);
  }

  inline int HitPattern::numberOfValidTimingBTLHits() const {
    return countTypedHits(TRACK_HITS, validHitFilter, timingBTLHitFilter);
  }

  inline int HitPattern::numberOfValidTimingETLHits() const {
    return countTypedHits(TRACK_HITS, validHitFilter, timingETLHitFilter);
  }

  inline int HitPattern::numberOfLostHits(HitCategory category) const { return countHits(category, missingHitFilter); }

  inline int HitPattern::numberOfLostTrackerHits(HitCategory category) const {
    return countTypedHits(category, missingHitFilter, trackerHitFilter);
  }

  inline int HitPattern::numberOfLostMuonHits() const {
    return countTypedHits(TRACK_HITS, missingHitFilter, muonHitFilter);
  }

  inline int HitPattern::numberOfLostTimingHits() const {
    return countTypedHits(TRACK_HITS, missingHitFilter, timingHitFilter);
  }

  inline int HitPattern::numberOfLostTimingBTLHits() const {
    return countTypedHits(TRACK_HITS, missingHitFilter, timingBTLHitFilter);
  }

  inline int HitPattern::numberOfLostTimingETLHits() const {
    return countTypedHits(TRACK_HITS, missingHitFilter, timingETLHitFilter);
  }

  inline int HitPattern::numberOfLostPixelHits(HitCategory category) const {
    return countTypedHits(category, missingHitFilter, pixelHitFilter);
  }

  inline int HitPattern::numberOfLostPixelBarrelHits(HitCategory category) const {
    return countTypedHits(category, missingHitFilter, pixelBarrelHitFilter);
  }

  inline int HitPattern::numberOfLostPixelEndcapHits(HitCategory category) const {
    return countTypedHits(category, missingHitFilter, pixelEndcapHitFilter);
  }

  inline int HitPattern::numberOfLostStripHits(HitCategory category) const {
    return countTypedHits(category, missingHitFilter, stripHitFilter);
  }

  inline int HitPattern::numberOfLostStripTIBHits(HitCategory category) const {
    return countTypedHits(category, missingHitFilter, stripTIBHitFilter);
  }

  inline int HitPattern::numberOfLostStripTIDHits(HitCategory category) const {
    return countTypedHits(category, missingHitFilter, stripTIDHitFilter);
  }

  inline int HitPattern::numberOfLostStripTOBHits(HitCategory category) const {
    return countTypedHits(category, missingHitFilter, stripTOBHitFilter);
  }

  inline int HitPattern::numberOfLostStripTECHits(HitCategory category) const {
    return countTypedHits(category, missingHitFilter, stripTECHitFilter);
  }

  inline int HitPattern::numberOfLostMuonDTHits() const {
    return countTypedHits(TRACK_HITS, missingHitFilter, muonDTHitFilter);
  }

  inline int HitPattern::numberOfLostMuonCSCHits() const {
    return countTypedHits(TRACK_HITS, missingHitFilter, muonCSCHitFilter);
  }

  inline int HitPattern::numberOfLostMuonRPCHits() const {
    return countTypedHits(TRACK_HITS, missingHitFilter, muonRPCHitFilter);
  }

  inline int HitPattern::numberOfLostMuonGEMHits() const {
    return countTypedHits(TRACK_HITS, missingHitFilter, muonGEMHitFilter);
  }

  inline int HitPattern::numberOfLostMuonME0Hits() const {
    return countTypedHits(TRACK_HITS, missingHitFilter, muonME0HitFilter);
  }

  inline int HitPattern::numberOfBadHits() const { return countHits(TRACK_HITS, badHitFilter); }

  inline int HitPattern::numberOfBadMuonHits() const {
    return countTypedHits(TRACK_HITS, inactiveHitFilter, muonHitFilter);
  }

  inline int HitPattern::numberOfBadMuonDTHits() const {
    return countTypedHits(TRACK_HITS, inactiveHitFilter, muonDTHitFilter);
  }

  inline int HitPattern::numberOfBadMuonCSCHits() const {
    return countTypedHits(TRACK_HITS, inactiveHitFilter, muonCSCHitFilter);
  }

  inline int HitPattern::numberOfBadMuonRPCHits() const {
    return countTypedHits(TRACK_HITS, inactiveHitFilter, muonRPCHitFilter);
  }

  inline int HitPattern::numberOfBadMuonGEMHits() const {
    return countTypedHits(TRACK_HITS, inactiveHitFilter, muonGEMHitFilter);
  }

  inline int HitPattern::numberOfBadMuonME0Hits() const {
    return countTypedHits(TRACK_HITS, inactiveHitFilter, muonME0HitFilter);
  }

  inline int HitPattern::numberOfInactiveHits() const { return countHits(TRACK_HITS, inactiveHitFilter); }

  inline int HitPattern::numberOfInactiveTrackerHits() const {
    return countTypedHits(TRACK_HITS, inactiveHitFilter, trackerHitFilter);
  }

  inline int HitPattern::trackerLayersWithMeasurementOld() const {
    return pixelLayersWithMeasurement() + stripLayersWithMeasurement();
  }

  inline int HitPattern::pixelLayersWithMeasurementOld() const {
    return pixelBarrelLayersWithMeasurement() + pixelEndcapLayersWithMeasurement();
  }

  inline int HitPattern::stripLayersWithMeasurement() const {
    return stripTIBLayersWithMeasurement() + stripTIDLayersWithMeasurement() + stripTOBLayersWithMeasurement() +
           stripTECLayersWithMeasurement();
  }

  inline int HitPattern::trackerLayersWithoutMeasurementOld(HitCategory category) const {
    return pixelLayersWithoutMeasurement(category) + stripLayersWithoutMeasurement(category);
  }

  inline int HitPattern::pixelLayersWithoutMeasurement(HitCategory category) const {
    return pixelBarrelLayersWithoutMeasurement(category) + pixelEndcapLayersWithoutMeasurement(category);
  }

  inline int HitPattern::stripLayersWithoutMeasurement(HitCategory category) const {
    return stripTIBLayersWithoutMeasurement(category) + stripTIDLayersWithoutMeasurement(category) +
           stripTOBLayersWithoutMeasurement(category) + stripTECLayersWithoutMeasurement(category);
  }

  inline int HitPattern::trackerLayersTotallyOffOrBad(HitCategory category) const {
    return pixelLayersTotallyOffOrBad(category) + stripLayersTotallyOffOrBad(category);
  }

  inline int HitPattern::pixelLayersTotallyOffOrBad(HitCategory category) const {
    return pixelBarrelLayersTotallyOffOrBad(category) + pixelEndcapLayersTotallyOffOrBad(category);
  }

  inline int HitPattern::stripLayersTotallyOffOrBad(HitCategory category) const {
    return stripTIBLayersTotallyOffOrBad(category) + stripTIDLayersTotallyOffOrBad(category) +
           stripTOBLayersTotallyOffOrBad(category) + stripTECLayersTotallyOffOrBad(category);
  }

  inline int HitPattern::trackerLayersNull() const { return pixelLayersNull() + stripLayersNull(); }

  inline int HitPattern::pixelLayersNull() const { return pixelBarrelLayersNull() + pixelEndcapLayersNull(); }

  inline int HitPattern::stripLayersNull() const {
    return stripTIBLayersNull() + stripTIDLayersNull() + stripTOBLayersNull() + stripTECLayersNull();
  }

  inline int HitPattern::muonStationsWithValidHits() const { return muonStations(0, 0); }

  inline int HitPattern::muonStationsWithBadHits() const { return muonStations(0, 3); }

  inline int HitPattern::muonStationsWithAnyHits() const { return muonStations(0, -1); }

  inline int HitPattern::dtStationsWithValidHits() const { return muonStations(1, 0); }

  inline int HitPattern::dtStationsWithBadHits() const { return muonStations(1, 3); }

  inline int HitPattern::dtStationsWithAnyHits() const { return muonStations(1, -1); }

  inline int HitPattern::cscStationsWithValidHits() const { return muonStations(2, 0); }

  inline int HitPattern::cscStationsWithBadHits() const { return muonStations(2, 3); }

  inline int HitPattern::cscStationsWithAnyHits() const { return muonStations(2, -1); }

  inline int HitPattern::rpcStationsWithValidHits() const { return muonStations(3, 0); }

  inline int HitPattern::rpcStationsWithBadHits() const { return muonStations(3, 3); }

  inline int HitPattern::rpcStationsWithAnyHits() const { return muonStations(3, -1); }

  inline int HitPattern::gemStationsWithValidHits() const { return muonStations(4, 0); }

  inline int HitPattern::gemStationsWithBadHits() const { return muonStations(4, 3); }

  inline int HitPattern::gemStationsWithAnyHits() const { return muonStations(4, -1); }

  inline int HitPattern::me0StationsWithValidHits() const { return muonStations(5, 0); }

  inline int HitPattern::me0StationsWithBadHits() const { return muonStations(5, 3); }

  inline int HitPattern::me0StationsWithAnyHits() const { return muonStations(5, -1); }

  inline int HitPattern::innermostMuonStationWithValidHits() const { return innermostMuonStationWithHits(0); }

  inline int HitPattern::innermostMuonStationWithBadHits() const { return innermostMuonStationWithHits(3); }

  inline int HitPattern::innermostMuonStationWithAnyHits() const { return innermostMuonStationWithHits(-1); }

  inline int HitPattern::outermostMuonStationWithValidHits() const { return outermostMuonStationWithHits(0); }

  inline int HitPattern::outermostMuonStationWithBadHits() const { return outermostMuonStationWithHits(3); }

  inline int HitPattern::outermostMuonStationWithAnyHits() const { return outermostMuonStationWithHits(-1); }

  template <int N = HitPattern::MaxHits>
  struct PatternSet {
    static constexpr int MaxHits = N;
    unsigned char hit[N];
    unsigned char nhit;

    unsigned char const *begin() const { return hit; }

    unsigned char const *end() const { return hit + nhit; }

    unsigned char *begin() { return hit; }

    unsigned char *end() { return hit + nhit; }

    int size() const { return nhit; }

    unsigned char operator[](int i) const { return hit[i]; }

    PatternSet() : nhit(0) {}

    PatternSet(HitPattern::HitCategory category, HitPattern const &hp) { fill(category, hp); }

    void fill(HitPattern::HitCategory category, HitPattern const &hp) {
      int lhit = 0;
      auto unpack = [&lhit, this](uint16_t pattern) -> bool {
        unsigned char p = 255 & (pattern >> 3);
        hit[lhit++] = p;

        // bouble sort
        if (lhit > 1) {
          for (auto h = hit + lhit - 1; h != hit; --h) {
            if ((*(h - 1)) <= p) {
              break;
            }
            (*h) = *(h - 1);
            *(h - 1) = p;
          }
        }
        return lhit < MaxHits;
      };

      hp.call(category, HitPattern::validHitFilter, unpack);
      nhit = lhit;
    }
  };

  template <int N>
  inline PatternSet<N> commonHits(PatternSet<N> const &p1, PatternSet<N> const &p2) {
    PatternSet<N> comm;
    comm.nhit = std::set_intersection(p1.begin(), p1.end(), p2.begin(), p2.end(), comm.begin()) - comm.begin();
    return comm;
  }

}  // namespace reco

#endif
