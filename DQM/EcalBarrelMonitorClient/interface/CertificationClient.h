#ifndef CertificationClient_H
#define CertificationClient_H

#include "DQM/EcalCommon/interface/DQWorkerClient.h"

namespace ecaldqm {

  class CertificationClient : public DQWorkerClient {
  public:
    CertificationClient(const edm::ParameterSet &, const edm::ParameterSet &);
    ~CertificationClient() {}

    void bookMEs();

    void beginRun(const edm::Run &, const edm::EventSetup &);

    void producePlots();

    enum MESets {
      kCertificationMap,
      kCertificationContents,
      kCertification,
      kReportSummaryMap,
      kReportSummaryContents,
      kReportSummary,
      nMESets
    };

    static void setMEData(std::vector<MEData>&);

    enum Sources {
      sIntegrity,
      sFEStatus,
      sDesync,
      sDAQ,
      sDCS,
      nSources
    };
  };

}

#endif

