#ifndef MTCCHLTrigger_H
#define MTCCHLTrigger_H

#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace cms {
  class MTCCHLTrigger : public edm::stream::EDFilter<> {
  public:
    MTCCHLTrigger(const edm::ParameterSet& ps);
    ~MTCCHLTrigger() override = default;

    bool filter(edm::Event& e, edm::EventSetup const& c) override;

  private:
    bool selOnDigiCharge;
    unsigned int ChargeThreshold;
    std::string clusterProducer;
  };
}  // namespace cms
#endif
