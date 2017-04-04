#ifndef L1TMuonEndCap_SectorProcessor_hh
#define L1TMuonEndCap_SectorProcessor_hh

#include <deque>
#include <map>
#include <string>
#include <vector>

#include "L1Trigger/L1TMuonEndCap/interface/Common.hh"

#include "L1Trigger/L1TMuonEndCap/interface/SectorProcessorLUT.hh"
#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngine.hh"

#include "L1Trigger/L1TMuonEndCap/interface/PrimitiveSelection.hh"
#include "L1Trigger/L1TMuonEndCap/interface/PrimitiveConversion.hh"
#include "L1Trigger/L1TMuonEndCap/interface/PatternRecognition.hh"
#include "L1Trigger/L1TMuonEndCap/interface/PrimitiveMatching.hh"
#include "L1Trigger/L1TMuonEndCap/interface/AngleCalculation.hh"
#include "L1Trigger/L1TMuonEndCap/interface/BestTrackSelection.hh"
#include "L1Trigger/L1TMuonEndCap/interface/PtAssignment.hh"
#include "L1Trigger/L1TMuonEndCap/interface/SingleHitTrack.hh"


class SectorProcessor {
public:
  explicit SectorProcessor();
  ~SectorProcessor();

  typedef unsigned long long EventNumber_t;
  typedef PatternRecognition::pattern_ref_t pattern_ref_t;

  void configure(
      const GeometryTranslator* tp_geom,
      const SectorProcessorLUT* lut,
      const PtAssignmentEngine* pt_assign_engine,
      int verbose, int endcap, int sector,
      int minBX, int maxBX, int bxWindow, int bxShiftCSC, int bxShiftRPC,
      const std::vector<int>& zoneBoundaries, int zoneOverlap, int zoneOverlapRPC,
      bool includeNeighbor, bool duplicateTheta, bool fixZonePhi, bool useNewZones, bool fixME11Edges,
      const std::vector<std::string>& pattDefinitions, const std::vector<std::string>& symPattDefinitions, bool useSymPatterns,
      int thetaWindow, int thetaWindowRPC, bool bugME11Dupes,
      int maxRoadsPerZone, int maxTracks, bool useSecondEarliest, bool bugSameSectorPt0,
      bool readPtLUTFile, bool fixMode15HighPt, bool bug9BitDPhi, bool bugMode7CLCT, bool bugNegPt, bool bugGMTPhi
  );

  void process(
      // Input
      EventNumber_t ievent,
      const TriggerPrimitiveCollection& muon_primitives,
      // Output
      EMTFHitCollection& out_hits,
      EMTFTrackCollection& out_tracks
  ) const;

  void process_single_bx(
      // Input
      int bx,
      const TriggerPrimitiveCollection& muon_primitives,
      // Output
      EMTFHitCollection& out_hits,
      EMTFTrackCollection& out_tracks,
      // Intermediate objects
      std::deque<EMTFHitCollection>& extended_conv_hits,
      std::deque<EMTFTrackCollection>& extended_best_track_cands,
      std::map<pattern_ref_t, int>& patt_lifetime_map
  ) const;

private:
  const GeometryTranslator* tp_geom_;

  const SectorProcessorLUT* lut_;

  const PtAssignmentEngine* pt_assign_engine_;

  int verbose_, endcap_, sector_;

  int minBX_, maxBX_, bxWindow_, bxShiftCSC_, bxShiftRPC_;

  // For primitive conversion
  std::vector<int> zoneBoundaries_;
  int zoneOverlap_, zoneOverlapRPC_;
  bool includeNeighbor_, duplicateTheta_, fixZonePhi_, useNewZones_, fixME11Edges_;

  // For pattern recognition
  std::vector<std::string> pattDefinitions_, symPattDefinitions_;
  bool useSymPatterns_;

  // For track building
  int thetaWindow_, thetaWindowRPC_;
  bool bugME11Dupes_;

  // For ghost cancellation
  int maxRoadsPerZone_, maxTracks_;
  bool useSecondEarliest_;
  bool bugSameSectorPt0_;

  // For pt assignment
  bool readPtLUTFile_, fixMode15HighPt_;
  bool bug9BitDPhi_, bugMode7CLCT_, bugNegPt_, bugGMTPhi_;
};

#endif
