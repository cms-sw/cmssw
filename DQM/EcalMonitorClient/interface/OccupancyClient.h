#ifndef OccupancyClient_H
#define OccupancyClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class OccupancyClient : public DQWorkerClient {
  public:
    OccupancyClient();
    ~OccupancyClient() override {}

    void producePlots(ProcessType) override;

  private:
    void setParams(edm::ParameterSet const&) override;

    int minHits_;
    float deviationThreshold_;
  };

}  // namespace ecaldqm

#endif
