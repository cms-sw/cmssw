#ifndef L1Trigger_L1TMuonEndCapPhase2_RPCTPSelector_h
#define L1Trigger_L1TMuonEndCapPhase2_RPCTPSelector_h

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/TPrimitives.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/TPSelectors.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/Utils/RPCUtils.h"

namespace emtf::phase2 {

  class RPCTPSelector : public TPSelector {
  public:
    explicit RPCTPSelector(const EMTFContext&, const int&, const int&);

    ~RPCTPSelector() override = default;

    void select(const TriggerPrimitive&, TPInfo, ILinkTPCMap&) const final;

  private:
    const EMTFContext& context_;

    int endcap_, sector_;

    int get_input_link(const TriggerPrimitive&, TPInfo&) const;

    int calculate_input_link(const int&, const int&, const int&, const int&, const TPSelection&) const;
  };

}  // namespace emtf::phase2

#endif  // L1Trigger_L1TMuonEndCapPhase2_RPCTPSelector_h
