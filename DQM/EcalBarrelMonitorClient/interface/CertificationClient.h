#ifndef CertificationClient_H
#define CertificationClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class CertificationClient : public DQWorkerClient {
  public:
    CertificationClient(edm::ParameterSet const&);
    ~CertificationClient() {}

    void bookMEs();

    void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

    void producePlots();

    enum MESets {
      kCertificationMap,
      kCertificationContents,
      kCertification,
      nTargets,
      sIntegrity = 0,
      sFEStatus,
      sDesync,
      sDAQ,
      sDCS,
      sReport,
      nSources,
      nMESets = nTargets + nSources
    };

    static void setMEData(std::vector<MEData>&);
  };

}

#endif

