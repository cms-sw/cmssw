#ifndef L1TMuonEndCap_Phase2SectorProcessor_h_experimental
#define L1TMuonEndCap_Phase2SectorProcessor_h_experimental


// _____________________________________________________________________________
// This implements a TEMPORARY version of the Phase 2 EMTF sector processor.
// It is supposed to be replaced in the future. It is intentionally written
// in a monolithic fashion to allow easy replacement.
//

#include <algorithm>
#include <array>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/L1TMuonEndCap/interface/Common.h"
//#include "L1Trigger/L1TMuonEndCap/interface/GeometryTranslator.h"
#include "L1Trigger/L1TMuonEndCap/interface/ConditionHelper.h"
#include "L1Trigger/L1TMuonEndCap/interface/SectorProcessorLUT.h"
#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngine.h"
//#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngine2016.h"
//#include "L1Trigger/L1TMuonEndCap/interface/PtAssignmentEngine2017.h"

#include "L1Trigger/L1TMuonEndCap/interface/PrimitiveSelection.h"
#include "L1Trigger/L1TMuonEndCap/interface/PrimitiveConversion.h"


namespace experimental {

class Hit;   // internal class
class Road;  // internal class
class Track; // internal class

class Phase2SectorProcessor {
public:
  void configure(
      // Object pointers
      const GeometryTranslator* geom,
      const ConditionHelper* cond,
      const SectorProcessorLUT* lut,
      PtAssignmentEngine* pt_assign_engine,
      // Sector processor config
      int verbose, int endcap, int sector, int bx,
      int bxShiftCSC, int bxShiftRPC, int bxShiftGEM,
      std::string era
  );

  void process(
      // Input
      const edm::Event& iEvent, const edm::EventSetup& iSetup,
      const TriggerPrimitiveCollection& muon_primitives,
      // Output
      EMTFHitCollection& out_hits,
      EMTFTrackCollection& out_tracks
  ) const;

private:
  void build_tracks(
      // Input
      const EMTFHitCollection& conv_hits,
      // Output
      std::vector<Track>& best_tracks
  ) const;

  void convert_tracks(
      // Input
      const EMTFHitCollection& conv_hits,
      const std::vector<Track>& best_tracks,
      // Output
      EMTFTrackCollection& best_emtf_tracks
  ) const;

  void debug_tracks(
      // Input
      const std::vector<Hit>& hits,
      const std::vector<Road>& roads,
      const std::vector<Road>& clean_roads,
      const std::vector<Road>& slim_roads,
      const std::vector<Track>& tracks
  ) const;

  const GeometryTranslator* geom_;

  const ConditionHelper* cond_;

  const SectorProcessorLUT* lut_;

  PtAssignmentEngine* pt_assign_engine_;

  int verbose_, endcap_, sector_, bx_,
      bxShiftCSC_, bxShiftRPC_, bxShiftGEM_;

  std::string era_;
};

}  // namesapce experimental

#endif  // L1TMuonEndCap_Phase2SectorProcessor_h_experimental
