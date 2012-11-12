#ifndef PresampleClient_H
#define PresampleClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class PresampleClient : public DQWorkerClient {
  public:
    PresampleClient(edm::ParameterSet const&, edm::ParameterSet const&);
    ~PresampleClient() {}

    void producePlots();

  protected:
    int minChannelEntries_;
    float expectedMean_;
    float toleranceMean_;
    float toleranceRMS_;
    float toleranceRMSFwd_;
  };

}

#endif

