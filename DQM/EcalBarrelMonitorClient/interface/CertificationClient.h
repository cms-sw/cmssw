#ifndef CertificationClient_H
#define CertificationClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class CertificationClient : public DQWorkerClient {
  public:
    CertificationClient(edm::ParameterSet const&, edm::ParameterSet const&);
    ~CertificationClient() {}

    void beginRun(edm::Run const&, edm::EventSetup const&);

    void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

    void producePlots();

    enum MESets {
      kCertificationMap,
      kCertificationContents,
      kCertification,
      nMESets
    };

    enum Sources {
      kDAQ,
      kDCS,
      kReport,
      nSources
    };

    static void setMEOrdering(std::map<std::string, unsigned>&);
  };

}

#endif

