#ifndef TrigPrimClient_H
#define TrigPrimClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class TrigPrimClient : public DQWorkerClient {
  public:
    TrigPrimClient(edm::ParameterSet const&, edm::ParameterSet const&);
    ~TrigPrimClient() {}

    void beginRun(const edm::Run &, const edm::EventSetup &);

    void producePlots();

    enum MESets {
      //      kTiming,
      kTimingSummary,
      kNonSingleSummary,
      kEmulQualitySummary,
      nMESets
    };

    enum Sources {
      kEtRealMap,
      kEtEmulError,
      kMatchedIndex,
      nSources
    };

    static void setMEOrdering(std::map<std::string, unsigned>&);

  };

}

#endif

