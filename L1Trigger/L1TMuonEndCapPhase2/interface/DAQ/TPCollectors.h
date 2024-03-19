#ifndef L1Trigger_L1TMuonEndCapPhase2_TPCollectors_h
#define L1Trigger_L1TMuonEndCapPhase2_TPCollectors_h

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"

namespace emtf::phase2 {

  class TPCollector {
  public:
    TPCollector() = default;

    virtual ~TPCollector() = default;

    // Collects all the trigger primitives in the event
    virtual void collect(const edm::Event&, BXTPCMap&) const = 0;
  };

}  // namespace emtf::phase2

#endif  // L1Trigger_L1TMuonEndCapPhase2_TPCollectors_h
