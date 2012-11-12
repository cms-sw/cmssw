#ifndef TimingClient_H
#define TimingClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class TimingClient : public DQWorkerClient {
  public:
    TimingClient(edm::ParameterSet const&, edm::ParameterSet const&);
    ~TimingClient() {}

    void producePlots();

  protected:
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

