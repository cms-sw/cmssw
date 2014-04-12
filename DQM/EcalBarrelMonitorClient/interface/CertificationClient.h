#ifndef CertificationClient_H
#define CertificationClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class CertificationClient : public DQWorkerClient {
  public:
    CertificationClient();
    ~CertificationClient() {}

    void producePlots(ProcessType) override;
  };

}

#endif

