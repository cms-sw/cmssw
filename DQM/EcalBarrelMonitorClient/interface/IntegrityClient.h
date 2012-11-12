#ifndef IntegrityClient_H
#define IntegrityClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class IntegrityClient : public DQWorkerClient {
  public:
    IntegrityClient(edm::ParameterSet const&, edm::ParameterSet const&);
    ~IntegrityClient() {}

    void producePlots();

  protected:
    float errFractionThreshold_;
  };

}

#endif

