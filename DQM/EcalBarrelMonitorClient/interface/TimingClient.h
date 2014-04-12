#ifndef TimingClient_H
#define TimingClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class TimingClient : public DQWorkerClient {
  public:
    TimingClient();
    ~TimingClient() {}

    void producePlots(ProcessType) override;

  private:
    void setParams(edm::ParameterSet const&) override;

    float toleranceMean_;
    float toleranceMeanFwd_;
    float toleranceRMS_;
    float toleranceRMSFwd_;
    int minChannelEntries_;
    int minChannelEntriesFwd_;
    int minTowerEntries_;
    int minTowerEntriesFwd_;
    float tailPopulThreshold_;
  };

}

#endif

