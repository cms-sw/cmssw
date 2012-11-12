#ifndef OccupancyClient_H
#define OccupancyClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class OccupancyClient : public DQWorkerClient {
  public:
    OccupancyClient(edm::ParameterSet const&, edm::ParameterSet const&);
    ~OccupancyClient() {}

    void producePlots();

  private:
    int minHits_;
    float deviationThreshold_;
  };

}

#endif

