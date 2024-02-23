#ifndef L1Trigger_L1TMuonEndCapPhase2_GE0TPSelector_h
#define L1Trigger_L1TMuonEndCapPhase2_GE0TPSelector_h

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/TPrimitives.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/TPSelectors.h"

namespace emtf::phase2 {

  class GE0TPSelector : public TPSelector {
  public:
    explicit GE0TPSelector(const EMTFContext&, const int&, const int&);

    ~GE0TPSelector() override = default;

    void select(const TriggerPrimitive&, TPInfo, ILinkTPCMap&) const final;

  private:
    const EMTFContext& context_;

    int endcap_, sector_;

    int get_input_link(const TriggerPrimitive&, TPInfo&) const;

    int calculate_input_link(const int&, const int&, const TPSelection&) const;
  };

}  // namespace emtf::phase2

#endif  // L1Trigger_L1TMuonEndCapPhase2_GE0TPSelector_h
