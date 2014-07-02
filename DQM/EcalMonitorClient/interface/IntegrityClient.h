#ifndef IntegrityClient_H
#define IntegrityClient_H

#include "DQWorkerClient.h"

namespace ecaldqm
{
  class IntegrityClient : public DQWorkerClient {
  public:
    IntegrityClient();
    ~IntegrityClient() {}

    void producePlots(ProcessType) override;

  private:
    void setParams(edm::ParameterSet const&) override;

    float errFractionThreshold_;
  };
}

#endif

