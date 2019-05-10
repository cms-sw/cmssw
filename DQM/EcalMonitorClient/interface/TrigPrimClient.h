#ifndef TrigPrimClient_H
#define TrigPrimClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {
  class TrigPrimClient : public DQWorkerClient {
  public:
    TrigPrimClient();
    ~TrigPrimClient() override {}

    void producePlots(ProcessType) override;

  private:
    void setParams(edm::ParameterSet const&) override;

    int minEntries_;
    float errorFractionThreshold_;
    float TTF4MaskingAlarmThreshold_;
  };

}  // namespace ecaldqm

#endif
