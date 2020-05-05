#ifndef L1TMuonEndCap_SectorProcessor_h
#define L1TMuonEndCap_SectorProcessor_h

#include <deque>
#include <map>
#include <string>
#include <vector>

#include "DataFormats/Provenance/interface/EventID.h"

#include "L1Trigger/L1TMuonEndCap/interface/Common.h"
#include "L1Trigger/L1TMuonEndCap/interface/EMTFSetup.h"

#include "L1Trigger/L1TMuonEndCap/interface/PrimitiveSelection.h"
#include "L1Trigger/L1TMuonEndCap/interface/PrimitiveConversion.h"
#include "L1Trigger/L1TMuonEndCap/interface/PatternRecognition.h"
#include "L1Trigger/L1TMuonEndCap/interface/PrimitiveMatching.h"
#include "L1Trigger/L1TMuonEndCap/interface/AngleCalculation.h"
#include "L1Trigger/L1TMuonEndCap/interface/BestTrackSelection.h"
#include "L1Trigger/L1TMuonEndCap/interface/PtAssignment.h"
#include "L1Trigger/L1TMuonEndCap/interface/SingleHitTrack.h"

class SectorProcessor {
public:
  explicit SectorProcessor();
  ~SectorProcessor();

  typedef PatternRecognition::pattern_ref_t pattern_ref_t;

  void configure(const EMTFSetup* setup, int verbose, int endcap, int sector);

  void process(
      // Input
      const edm::EventID& event_id,
      const TriggerPrimitiveCollection& muon_primitives,
      // Output
      EMTFHitCollection& out_hits,
      EMTFTrackCollection& out_tracks) const;

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
      std::map<pattern_ref_t, int>& patt_lifetime_map) const;

private:
  const EMTFSetup* setup_;

  int verbose_, endcap_, sector_;
};

#endif
