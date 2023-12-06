#ifndef L1Trigger_L1TMuonEndCapPhase2_EMTFConstants_h
#define L1Trigger_L1TMuonEndCapPhase2_EMTFConstants_h

namespace emtf::phase2 {
    // from DataFormats/MuonDetId/interface/CSCDetId.h
    constexpr int kMinEndcap = 1;
    constexpr int kMaxEndcap = 2;
    constexpr int kMinStation = 1;
    constexpr int kMaxStation = 4;
    constexpr int kMinRing = 1;
    constexpr int kMaxRing = 4;
    constexpr int kMinChamber = 1;
    constexpr int kMaxChamber = 36;
    constexpr int kMinLayer = 1;
    constexpr int kMaxLayer = 6;

    // from DataFormats/MuonDetId/interface/CSCTriggerNumbering.h
    constexpr int kMinCSCId = 1;
    constexpr int kMaxCSCId = 9;
    constexpr int kMinTrigSector = 1;
    constexpr int kMaxTrigSector = 6;
    constexpr int kNumTrigSector = 12;
    constexpr int kMinTrigSubsector = 0;
    constexpr int kMaxTrigSubsector = 2;

    // Algorithm
    namespace v3 {
        constexpr int kNumChambers = 115;                                               // per sector
        constexpr int kChamberSegments = 2;                                             // per chamber
        constexpr int kNumSegments = kNumChambers * kChamberSegments;
        constexpr int kNumSegmentVariables = 13;                                        // per segment

        constexpr int kNumZones = 3;                                                    // per sector
        constexpr int kNumZonePatterns = 7;                                             // per zone

        constexpr int kNumTimeZones = 3;                                                // per sector

        constexpr int kNumTracks = 4;                                                   // per sector
        constexpr int kNumTrackVariables = 54;                                          // per track
        constexpr int kNumTrackFeatures = 40;                                           // per track
        constexpr int kNumTrackPredictions = 1;                                         // per track
        constexpr int kNumTrackSites = 12;                                              // per track
        constexpr int kNumTrackSitesRM = 5;                                             // per track

        constexpr int kChamberHitmapBW = 90;                                            // 24 deg
        constexpr int kChamberHitmapJoinedBW = 315;                                     // 84 deg
        constexpr int kHitmapNRows = 8;
        constexpr int kHitmapNCols = 288;
        constexpr int kHitmapNGates = 3;
        constexpr int kHitmapColFactor = 16;
        constexpr int kHitmapColFactorLog2 = 4;                                         // (1 << 4) = 16
        constexpr int kHitmapCropColStart = kChamberHitmapJoinedBW - kHitmapNCols;      // 27  (Inclusive)
        constexpr int kHitmapCropColStop = kChamberHitmapJoinedBW;                      // 315 (Exclusive)

        constexpr int kPatternNCols = 110;
        constexpr int kPatternMatchingPadding = 55;
        constexpr int kMaxPatternActivation = 63;
        constexpr int kMaxPatternActivationLog2 = 6;                                    // (1 << 6) - 1 = 63
    }
}

#endif  // L1Trigger_L1TMuonEndCapPhase2_EMTFConstants_h
