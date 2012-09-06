#ifndef SummaryClient_H
#define SummaryClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class SummaryClient : public DQWorkerClient {
  public:
    SummaryClient(edm::ParameterSet const&, edm::ParameterSet const&);
    ~SummaryClient() {}

    void beginRun(const edm::Run &, const edm::EventSetup &);

    void producePlots();

    enum MESets {
      kQualitySummary,
      kReportSummaryMap,
      kReportSummaryContents,
      kReportSummary,
      nMESets
    };

    enum Sources {
      kIntegrity,
      kPresample,
      kTiming,
      kRawData,
      kTriggerPrimitives,
      kHotCell,
      nSources
    };

    static void setMEOrdering(std::map<std::string, unsigned>&);

  protected:
    bool online_;
  };

}

#endif

