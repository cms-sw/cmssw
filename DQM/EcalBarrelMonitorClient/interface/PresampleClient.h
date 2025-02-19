#ifndef PresampleClient_H
#define PresampleClient_H

#include "DQM/EcalCommon/interface/DQWorkerClient.h"

namespace ecaldqm {

  class PresampleClient : public DQWorkerClient {
  public:
    PresampleClient(const edm::ParameterSet &, const edm::ParameterSet &);
    ~PresampleClient() {}

    void bookMEs();

    void producePlots();

    enum MESets {
      kQuality,
      kMean,
      kMeanDCC,
      kRMS,
      kRMSMap,
      kRMSMapSummary,
      kQualitySummary,
      nMESets
    };

    static void setMEData(std::vector<MEData>&);

    enum Sources {
      sPedestal,
      nSources
    };

  protected:
    int minChannelEntries_;
    int minTowerEntries_;
    float expectedMean_;
    float meanThreshold_;
    float rmsThreshold_;
    float rmsThresholdHighEta_;
    float noisyFracThreshold_;
  };

}

#endif

