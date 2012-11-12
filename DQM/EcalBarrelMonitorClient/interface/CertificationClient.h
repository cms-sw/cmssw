#ifndef CertificationClient_H
#define CertificationClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class CertificationClient : public DQWorkerClient {
  public:
    CertificationClient(edm::ParameterSet const&, edm::ParameterSet const&);
    ~CertificationClient() {}

    void producePlots();
  };

}

#endif

