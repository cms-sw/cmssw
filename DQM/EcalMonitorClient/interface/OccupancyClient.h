#ifndef OccupancyClient_H
#define OccupancyClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class OccupancyClient : public DQWorkerClient {
  public:
    OccupancyClient();
    ~OccupancyClient() {}

    void producePlots(ProcessType) override;

  private:
    void setParams(edm::ParameterSet const&) override;

    int minHits_;
    float deviationThreshold_;
  };

}

#endif

