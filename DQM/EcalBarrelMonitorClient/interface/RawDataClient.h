#ifndef RawDataClient_H
#define RawDataClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class RawDataClient : public DQWorkerClient {
  public:
    RawDataClient(edm::ParameterSet const&, edm::ParameterSet const&);
    ~RawDataClient() {}

    void producePlots();

    enum MESets {
      kQualitySummary,
      nMESets
    };

    enum Sources {
      kEntries,
      kL1ADCC,
      kFEStatus,
      nSources
    };

    static void setMEOrdering(std::map<std::string, unsigned>&);

  private:
    float synchErrThresholdFactor_;
  };

}

#endif

