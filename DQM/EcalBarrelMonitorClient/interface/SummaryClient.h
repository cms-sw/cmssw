#ifndef SummaryClient_H
#define SummaryClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class SummaryClient : public DQWorkerClient {
  public:
    SummaryClient(edm::ParameterSet const&, edm::ParameterSet const&);
    ~SummaryClient() {}

    void bookMEs();

    void beginRun(const edm::Run &, const edm::EventSetup &);

    void producePlots();

    enum MESets {
      kQualitySummary,
      kReportSummaryMap,
      kReportSummaryContents,
      kReportSummary,
      kNBadFEDs,
      nMESets
    };

    enum Sources {
      kIntegrity,
      kIntegrityByLumi,
      kPresample,
      kTiming,
      kRawData,
      kDesyncByLumi,
      kFEByLumi,
      kTriggerPrimitives,
      kHotCell,
      nSources
    };

    static void setMEOrdering(std::map<std::string, unsigned>&);

  private:
    float towerBadFraction_;
    float fedBadFraction_;
  };

}

#endif

