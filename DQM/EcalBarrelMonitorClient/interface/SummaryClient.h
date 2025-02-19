#ifndef SummaryClient_H
#define SummaryClient_H

#include "DQM/EcalCommon/interface/DQWorkerClient.h"

namespace ecaldqm {

  class SummaryClient : public DQWorkerClient {
  public:
    SummaryClient(const edm::ParameterSet &, const edm::ParameterSet &);
    ~SummaryClient() {}

    void bookMEs();

    void beginRun(const edm::Run &, const edm::EventSetup &);

    void producePlots();

    enum MESets {
      kQualitySummary,
      kReportSummaryMap,
      kReportSummaryContents,
      kReportSummary,
      nMESets
    };

    static void setMEData(std::vector<MEData>&);

    enum Sources {
      sIntegrity,
      sPresample,
      sTiming,
      sRawData,
      sDigiOccupancy,
      nSources
    };
  };

}

#endif

