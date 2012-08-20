#ifndef RawDataClient_H
#define RawDataClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class RawDataClient : public DQWorkerClient {
  public:
    RawDataClient(const edm::ParameterSet &);
    ~RawDataClient() {}

    void bookMEs();

    void producePlots();

    enum MESets {
      kQualitySummary,
      nTargets,
      sL1ADCC = 0,
      sFEStatus,
      nSources,
      nMESets = nTargets + nSources
    };

    static void setMEData(std::vector<MEData>&);

  private:
    int synchErrorThreshold_;
  };

}

#endif

