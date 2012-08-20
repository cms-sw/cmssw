#ifndef SummaryClient_H
#define SummaryClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class SummaryClient : public DQWorkerClient {
  public:
    SummaryClient(const edm::ParameterSet &);
    ~SummaryClient() {}

    void bookMEs();

    void beginRun(const edm::Run &, const edm::EventSetup &);

    void producePlots();

    enum MESets {
      kQualitySummary,
      kReportSummaryMap,
      kReportSummaryContents,
      kReportSummary,
      nTargets,
      sIntegrity = 0,
      sPresample,
      sTiming,
      sRawData,
      sDigiOccupancy,
      nSources,
      nMESets = nTargets + nSources
    };

    static void setMEData(std::vector<MEData>&);

  };

}

#endif

