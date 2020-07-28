#ifndef IntegrityClient_H
#define IntegrityClient_H

#include "DQWorkerClient.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

namespace ecaldqm {
  class IntegrityClient : public DQWorkerClient {
  public:
    IntegrityClient();
    ~IntegrityClient() override {}

    void producePlots(ProcessType) override;

  private:
    void setParams(edm::ParameterSet const&) override;

    float errFractionThreshold_;
  };
}  // namespace ecaldqm

#endif
