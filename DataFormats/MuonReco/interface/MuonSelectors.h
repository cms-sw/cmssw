#ifndef MuonReco_MuonSelectors_h
#define MuonReco_MuonSelectors_h
//
// Package:    MuonReco
//
//
// Original Author:  Jake Ribnik, Dmytro Kovalskyi

#include "DataFormats/MuonReco/interface/Muon.h"
#include <string>

namespace reco {
  class Vertex;
}

namespace muon {
  /// Selector type
  enum SelectionType {
    All = 0,                                // dummy options - always true
    AllGlobalMuons = 1,                     // checks isGlobalMuon flag
    AllStandAloneMuons = 2,                 // checks isStandAloneMuon flag
    AllTrackerMuons = 3,                    // checks isTrackerMuon flag
    TrackerMuonArbitrated = 4,              // resolve ambiguity of sharing segments
    AllArbitrated = 5,                      // all muons with the tracker muon arbitrated
    GlobalMuonPromptTight = 6,              // global muons with tighter fit requirements
    TMLastStationLoose = 7,                 // penetration depth loose selector
    TMLastStationTight = 8,                 // penetration depth tight selector
    TM2DCompatibilityLoose = 9,             // likelihood based loose selector
    TM2DCompatibilityTight = 10,            // likelihood based tight selector
    TMOneStationLoose = 11,                 // require one well matched segment
    TMOneStationTight = 12,                 // require one well matched segment
    TMLastStationOptimizedLowPtLoose = 13,  // combination of TMLastStation and TMOneStation
    TMLastStationOptimizedLowPtTight = 14,  // combination of TMLastStation and TMOneStation
    GMTkChiCompatibility = 15,              // require tk stub have good chi2 relative to glb track
    GMStaChiCompatibility = 16,             // require sta stub have good chi2 compatibility relative to glb track
    GMTkKinkTight = 17,                     // require a small kink value in the tracker stub
    TMLastStationAngLoose = 18,             // TMLastStationLoose with additional angular cuts
    TMLastStationAngTight = 19,             // TMLastStationTight with additional angular cuts
    TMOneStationAngLoose = 20,              // TMOneStationLoose with additional angular cuts
    TMOneStationAngTight = 21,              // TMOneStationTight with additional angular cuts
    // The two algorithms that follow are identical to what were known as
    // TMLastStationOptimizedLowPt* (sans the Barrel) as late as revision
    // 1.7 of this file. The names were changed because indeed the low pt
    // optimization applies only to the barrel region, whereas the sel-
    // ectors above are more efficient at low pt in the endcaps, which is
    // what we feel is more suggestive of the algorithm name. This will be
    // less confusing for future generations of CMS members, I hope...
    // combination of TMLastStation and TMOneStation but with low pT optimization in barrel only
    TMLastStationOptimizedBarrelLowPtLoose = 22,
    // combination of TMLastStation and TMOneStation but with low pT optimization in barrel only
    TMLastStationOptimizedBarrelLowPtTight = 23,
    RPCMuLoose = 24,  // checks isRPCMuon flag (require two well matched hits in different RPC layers)
    AllME0Muons = 25,
    ME0MuonArbitrated = 26,
    AllGEMMuons = 27,
    GEMMuonArbitrated = 28,
    TriggerIdLoose = 29
  };

  /// a lightweight "map" for selection type string label and enum value
  struct SelectionTypeStringToEnum {
    const char* label;
    SelectionType value;
  };

  static const SelectionTypeStringToEnum selectionTypeStringToEnumMap[] = {
      {"All", All},
      {"AllGlobalMuons", AllGlobalMuons},
      {"AllStandAloneMuons", AllStandAloneMuons},
      {"AllTrackerMuons", AllTrackerMuons},
      {"TrackerMuonArbitrated", TrackerMuonArbitrated},
      {"AllArbitrated", AllArbitrated},
      {"GlobalMuonPromptTight", GlobalMuonPromptTight},
      {"TMLastStationLoose", TMLastStationLoose},
      {"TMLastStationTight", TMLastStationTight},
      {"TM2DCompatibilityLoose", TM2DCompatibilityLoose},
      {"TM2DCompatibilityTight", TM2DCompatibilityTight},
      {"TMOneStationLoose", TMOneStationLoose},
      {"TMOneStationTight", TMOneStationTight},
      {"TMLastStationOptimizedLowPtLoose", TMLastStationOptimizedLowPtLoose},
      {"TMLastStationOptimizedLowPtTight", TMLastStationOptimizedLowPtTight},
      {"GMTkChiCompatibility", GMTkChiCompatibility},
      {"GMStaChiCompatibility", GMStaChiCompatibility},
      {"GMTkKinkTight", GMTkKinkTight},
      {"TMLastStationAngLoose", TMLastStationAngLoose},
      {"TMLastStationAngTight", TMLastStationAngTight},
      {"TMOneStationAngLoose", TMOneStationAngLoose},
      {"TMOneStationAngTight", TMOneStationAngTight},
      {"TMLastStationOptimizedBarrelLowPtLoose", TMLastStationOptimizedBarrelLowPtLoose},
      {"TMLastStationOptimizedBarrelLowPtTight", TMLastStationOptimizedBarrelLowPtTight},
      {"RPCMuLoose", RPCMuLoose},
      {"AllME0Muons", AllME0Muons},
      {"ME0MuonArbitrated", ME0MuonArbitrated},
      {"AllGEMMuons", AllGEMMuons},
      {"GEMMuonArbitrated", GEMMuonArbitrated},
      {"TriggerIdLoose", TriggerIdLoose},
      {nullptr, (SelectionType)-1}};

  SelectionType selectionTypeFromString(const std::string& label);

  // a map for string label to reco::Muon::Selector enum
  struct SelectorStringToEnum {
    const char* label;
    reco::Muon::Selector value;
  };

  static const SelectorStringToEnum selectorStringToEnumMap[] = {
      {"CutBasedIdLoose", reco::Muon::CutBasedIdLoose},
      {"CutBasedIdMedium", reco::Muon::CutBasedIdMedium},
      {"CutBasedIdMediumPrompt", reco::Muon::CutBasedIdMediumPrompt},
      {"CutBasedIdTight", reco::Muon::CutBasedIdTight},
      {"CutBasedIdGlobalHighPt", reco::Muon::CutBasedIdGlobalHighPt},
      {"CutBasedIdTrkHighPt", reco::Muon::CutBasedIdTrkHighPt},
      {"PFIsoVeryLoose", reco::Muon::PFIsoVeryLoose},
      {"PFIsoLoose", reco::Muon::PFIsoLoose},
      {"PFIsoMedium", reco::Muon::PFIsoMedium},
      {"PFIsoTight", reco::Muon::PFIsoTight},
      {"PFIsoVeryTight", reco::Muon::PFIsoVeryTight},
      {"TkIsoLoose", reco::Muon::TkIsoLoose},
      {"TkIsoTight", reco::Muon::TkIsoTight},
      {"SoftCutBasedId", reco::Muon::SoftCutBasedId},
      {"SoftMvaId", reco::Muon::SoftMvaId},
      {"MvaLoose", reco::Muon::MvaLoose},
      {"MvaMedium", reco::Muon::MvaMedium},
      {"MvaTight", reco::Muon::MvaTight},
      {"MiniIsoLoose", reco::Muon::MiniIsoLoose},
      {"MiniIsoMedium", reco::Muon::MiniIsoMedium},
      {"MiniIsoTight", reco::Muon::MiniIsoTight},
      {"MiniIsoVeryTight", reco::Muon::MiniIsoVeryTight},
      {"TriggerIdLoose", reco::Muon::TriggerIdLoose},
      {"InTimeMuon", reco::Muon::InTimeMuon},
      {"PFIsoVeryVeryTight", reco::Muon::PFIsoVeryVeryTight},
      {"MultiIsoLoose", reco::Muon::MultiIsoLoose},
      {"MultiIsoMedium", reco::Muon::MultiIsoMedium},
      {"PuppiIsoLoose", reco::Muon::PuppiIsoLoose},
      {"PuppiIsoMedium", reco::Muon::PuppiIsoMedium},
      {"PuppiIsoTight", reco::Muon::PuppiIsoTight},
      {"MvaVTight", reco::Muon::MvaVTight},
      {"MvaVVTight", reco::Muon::MvaVVTight},
      {"LowPtMvaLoose", reco::Muon::LowPtMvaLoose},
      {"LowPtMvaMedium", reco::Muon::LowPtMvaMedium},
      {nullptr, (reco::Muon::Selector)-1}};

  reco::Muon::Selector selectorFromString(const std::string& label);

  /// main GoodMuon wrapper call
  bool isGoodMuon(const reco::Muon& muon,
                  SelectionType type,
                  reco::Muon::ArbitrationType arbitrationType = reco::Muon::SegmentAndTrackArbitration);

  // ===========================================================================
  //                               Support functions
  //
  enum AlgorithmType { TMLastStation, TM2DCompatibility, TMOneStation, RPCMu, ME0Mu, GEMMu };

  // specialized GoodMuon functions called from main wrapper
  bool isGoodMuon(const reco::Muon& muon,
                  AlgorithmType type,
                  double minCompatibility,
                  reco::Muon::ArbitrationType arbitrationType);

  bool isGoodMuon(const reco::Muon& muon,
                  AlgorithmType type,
                  int minNumberOfMatches,
                  double maxAbsDx,
                  double maxAbsPullX,
                  double maxAbsDy,
                  double maxAbsPullY,
                  double maxChamberDist,
                  double maxChamberDistPull,
                  reco::Muon::ArbitrationType arbitrationType,
                  bool syncMinNMatchesNRequiredStationsInBarrelOnly = true,  //this is what we had originally
                  bool applyAlsoAngularCuts = false);

  bool isTightMuon(const reco::Muon&, const reco::Vertex&);
  bool isLooseMuon(const reco::Muon&);
  bool isMediumMuon(const reco::Muon&, bool run2016_hip_mitigation = false);
  bool isSoftMuon(const reco::Muon&, const reco::Vertex&, bool run2016_hip_mitigation = false);
  bool isHighPtMuon(const reco::Muon&, const reco::Vertex&);
  bool isTrackerHighPtMuon(const reco::Muon&, const reco::Vertex&);
  bool isLooseTriggerMuon(const reco::Muon&);

  // determine if station was crossed well withing active volume
  unsigned int RequiredStationMask(const reco::Muon& muon,
                                   double maxChamberDist,
                                   double maxChamberDistPull,
                                   reco::Muon::ArbitrationType arbitrationType);

  // ------------ method to return the calo compatibility for a track with matched muon info  ------------
  float caloCompatibility(const reco::Muon& muon);

  // ------------ method to calculate the segment compatibility for a track with matched muon info  ------------
  float segmentCompatibility(const reco::Muon& muon,
                             reco::Muon::ArbitrationType arbitrationType = reco::Muon::SegmentAndTrackArbitration);

  // Check if two muon trajectory overlap
  // The overlap is performed by comparing distance between two muon
  // trajectories if they cross the same muon chamber. Trajectories
  // overlap if distance/uncertainty is smaller than allowed pullX
  // and pullY
  bool overlap(const reco::Muon& muon1,
               const reco::Muon& muon2,
               double pullX = 1.0,
               double pullY = 1.0,
               bool checkAdjacentChambers = false);

  /// Determine the number of shared segments between two muons.
  /// Comparison is done using the segment references in the reco::Muon object.
  int sharedSegments(const reco::Muon& muon1,
                     const reco::Muon& muon2,
                     unsigned int segmentArbitrationMask = reco::MuonSegmentMatch::BestInChamberByDR);

  reco::Muon::Selector makeSelectorBitset(reco::Muon const& muon,
                                          reco::Vertex const* vertex = nullptr,
                                          bool run2016_hip_mitigation = false);
}  // namespace muon
#endif
