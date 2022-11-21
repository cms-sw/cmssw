#ifndef PresampleClient_H
#define PresampleClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {
  class PresampleClient : public DQWorkerClient {
  public:
    PresampleClient();
    ~PresampleClient() override {}

    void producePlots(ProcessType) override;

  private:
    void setParams(edm::ParameterSet const&) override;

    int minChannelEntries_;
    float expectedMean_;
    float toleranceLow_;
    float toleranceHigh_;
    float toleranceHighFwd_;
    float toleranceRMS_;
    float toleranceRMSFwd_;
  };

}  // namespace ecaldqm

#endif
