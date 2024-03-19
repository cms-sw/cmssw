#ifndef L1Trigger_L1TMuonEndCapPhase2_TPSelectors_h
#define L1Trigger_L1TMuonEndCapPhase2_TPSelectors_h

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"

namespace emtf::phase2 {

  class TPSelector {
  public:
    TPSelector() = default;

    virtual ~TPSelector() = default;

    virtual void select(const TriggerPrimitive& tp, TPInfo tp_info, ILinkTPCMap&) const = 0;
  };

}  // namespace emtf::phase2

#endif  // L1Trigger_L1TMuonEndCapPhase2_TPSelectors_h
