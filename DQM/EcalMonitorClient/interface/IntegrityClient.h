#ifndef IntegrityClient_H
#define IntegrityClient_H

#include "DQWorkerClient.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

namespace ecaldqm
{
  class IntegrityClient : public DQWorkerClient {
  public:
    IntegrityClient();
    ~IntegrityClient() {}

    void producePlots(ProcessType) override;
    void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  private:
    void setParams(edm::ParameterSet const&) override;
    edm::ESHandle<EcalChannelStatus> chStatus;

    float errFractionThreshold_;
  };
}

#endif

