#ifndef PNIntegrityClient_H
#define PNIntegrityClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class PNIntegrityClient : public DQWorkerClient {
  public:
    PNIntegrityClient(edm::ParameterSet const&, edm::ParameterSet const&);
    ~PNIntegrityClient() {}

    void producePlots();

  protected:
    float errFractionThreshold_;
  };

}

#endif

