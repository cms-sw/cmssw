#ifndef PresampleClient_H
#define PresampleClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class PresampleClient : public DQWorkerClient {
  public:
    PresampleClient(edm::ParameterSet const&, edm::ParameterSet const&);
    ~PresampleClient() {}

    void bookMEs();

    void producePlots();

    enum MESets {
      kQuality,
      kMean,
      kMeanDCC,
      kRMS,
      kRMSMap,
      kQualitySummary,
      nMESets
    };

    enum Sources {
      kPedestal,
      nSources
    };

    static void setMEOrdering(std::map<std::string, unsigned>&);

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

