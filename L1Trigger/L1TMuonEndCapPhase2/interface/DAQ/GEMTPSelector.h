#ifndef L1Trigger_L1TMuonEndCapPhase2_GEMTPSelector_h
#define L1Trigger_L1TMuonEndCapPhase2_GEMTPSelector_h

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/TPrimitives.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/TPSelectors.h"

namespace emtf::phase2 {

  class GEMTPSelector : public TPSelector {
  public:
    explicit GEMTPSelector(const EMTFContext&, const int&, const int&);

    ~GEMTPSelector() override = default;

    void select(const TriggerPrimitive&, TPInfo, ILinkTPCMap&) const final;

  private:
    const EMTFContext& context_;

    int endcap_, sector_;

    int getInputLink(const TriggerPrimitive&, TPInfo&) const;

    int calcInputLink(const int&, const int&, const int&, const int&, const TPSelection&) const;
  };

}  // namespace emtf::phase2

#endif  // L1Trigger_L1TMuonEndCapPhase2_GEMTPSelector_h
