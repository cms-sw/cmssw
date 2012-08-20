#ifndef TrigPrimClient_H
#define TrigPrimClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class TrigPrimClient : public DQWorkerClient {
  public:
    TrigPrimClient(const edm::ParameterSet &);
    ~TrigPrimClient() {}

    void bookMEs();

    void producePlots();

    enum MESets {
      //      kTiming,
      kTimingSummary,
      kNonSingleSummary,
      kEmulQualitySummary,
      nTargets,
      sEtRealMap = 0,
      sEtEmulError,
      sTimingError,
      sMatchedIndex,
      nSources,
      nMESets = nTargets + nSources
    };

    static void setMEData(std::vector<MEData>&);

  };

}

#endif

