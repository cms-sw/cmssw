#ifndef SummaryClient_H
#define SummaryClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class SummaryClient : public DQWorkerClient {
  public:
    SummaryClient(edm::ParameterSet const&, edm::ParameterSet const&);
    ~SummaryClient() {}

    void bookMEs();

    void producePlots();

  private:
    float towerBadFraction_;
    float fedBadFraction_;
  };

}

#endif

