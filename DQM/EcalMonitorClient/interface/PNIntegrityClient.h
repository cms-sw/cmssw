#ifndef PNIntegrityClient_H
#define PNIntegrityClient_H

#include "DQWorkerClient.h"

namespace ecaldqm
{
  class PNIntegrityClient : public DQWorkerClient {
  public:
    PNIntegrityClient();
    ~PNIntegrityClient() {}

    void producePlots(ProcessType) override;

  private:
    void setParams(edm::ParameterSet const&) override;

    float errFractionThreshold_;
  };
}

#endif

