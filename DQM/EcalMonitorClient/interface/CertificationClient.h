#ifndef CertificationClient_H
#define CertificationClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class CertificationClient : public DQWorkerClient {
  public:
    CertificationClient();
    ~CertificationClient() override {}

    void producePlots(ProcessType) override;
  };

}  // namespace ecaldqm

#endif
