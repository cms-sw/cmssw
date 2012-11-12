#ifndef TrigPrimClient_H
#define TrigPrimClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class TrigPrimClient : public DQWorkerClient {
  public:
    TrigPrimClient(edm::ParameterSet const&, edm::ParameterSet const&);
    ~TrigPrimClient() {}

    void producePlots();

  private:
    int minEntries_;
    float errorFractionThreshold_;
  };

}

#endif

