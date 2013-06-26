#ifndef TrigPrimClient_H
#define TrigPrimClient_H

#include "DQM/EcalCommon/interface/DQWorkerClient.h"

namespace ecaldqm {

  class TrigPrimClient : public DQWorkerClient {
  public:
    TrigPrimClient(const edm::ParameterSet &, const edm::ParameterSet &);
    ~TrigPrimClient() {}

    void bookMEs();

    void producePlots();

    enum MESets {
      //      kTiming,
      kTimingSummary,
      kNonSingleSummary,
      kEmulQualitySummary,
      nMESets
    };

    static void setMEData(std::vector<MEData>&);

    enum Sources {
      sEtRealMap,
      sEtEmulError,
      sTimingError,
      sMatchedIndex,
      nSources
    };
  };

}

#endif

