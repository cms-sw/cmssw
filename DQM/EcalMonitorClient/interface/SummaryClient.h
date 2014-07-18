#ifndef SummaryClient_H
#define SummaryClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class SummaryClient : public DQWorkerClient {
  public:
    SummaryClient();
    ~SummaryClient() {}

    void producePlots(ProcessType) override;

  private:
    void setParams(edm::ParameterSet const&) override;

    float towerBadFraction_;
    float fedBadFraction_;
  };

}

#endif

