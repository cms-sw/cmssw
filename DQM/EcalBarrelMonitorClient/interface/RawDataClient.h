#ifndef RawDataClient_H
#define RawDataClient_H

#include "DQM/EcalCommon/interface/DQWorkerClient.h"

namespace ecaldqm {

  class RawDataClient : public DQWorkerClient {
  public:
    RawDataClient(const edm::ParameterSet &, const edm::ParameterSet &);
    ~RawDataClient() {}

    void bookMEs();

    void producePlots();

    enum MESets {
      kQualitySummary,
      nMESets
    };

    static void setMEData(std::vector<MEData>&);

    enum Sources {
      sL1ADCC,
      sFEStatus,
      nSources
    };

  private:
    int synchErrorThreshold_;
  };

}

#endif

