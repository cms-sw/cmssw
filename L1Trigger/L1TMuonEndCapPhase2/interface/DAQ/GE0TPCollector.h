#ifndef L1Trigger_L1TMuonEndCapPhase2_GE0TPCollector_h
#define L1Trigger_L1TMuonEndCapPhase2_GE0TPCollector_h

#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFfwd.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/EMTFTypes.h"
#include "L1Trigger/L1TMuonEndCapPhase2/interface/DAQ/TPCollectors.h"

namespace emtf::phase2 {

  class GE0TPCollector : public TPCollector {
  public:
    explicit GE0TPCollector(const EMTFContext&, edm::ConsumesCollector&);

    ~GE0TPCollector() override = default;

    void collect(const edm::Event&, BXTPCMap&) const final;

  private:
    const EMTFContext& context_;

    const edm::EDGetToken input_token_;
  };

}  // namespace emtf::phase2

#endif  // L1Trigger_L1TMuonEndCapPhase2_GE0TPCollector_h
