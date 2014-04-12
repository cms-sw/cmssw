#ifndef RawDataClient_H
#define RawDataClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class RawDataClient : public DQWorkerClient {
  public:
    RawDataClient();
    ~RawDataClient() {}

    void producePlots(ProcessType) override;

  private:
    void setParams(edm::ParameterSet const&) override;

    float synchErrThresholdFactor_;
  };

}

#endif

