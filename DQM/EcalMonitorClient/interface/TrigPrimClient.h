#ifndef TrigPrimClient_H
#define TrigPrimClient_H

#include "DQWorkerClient.h"

namespace ecaldqm
{
  class TrigPrimClient : public DQWorkerClient {
  public:
    TrigPrimClient();
    ~TrigPrimClient() {}

    void producePlots(ProcessType) override;

  private:
    void setParams(edm::ParameterSet const&) override;

    int minEntries_;
    float errorFractionThreshold_;
  };

}

#endif

