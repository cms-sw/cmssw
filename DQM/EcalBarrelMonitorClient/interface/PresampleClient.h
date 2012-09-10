#ifndef PresampleClient_H
#define PresampleClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class PresampleClient : public DQWorkerClient {
  public:
    PresampleClient(edm::ParameterSet const&, edm::ParameterSet const&);
    ~PresampleClient() {}

    void beginRun(const edm::Run &, const edm::EventSetup &);

    void producePlots();

    enum MESets {
      kQuality,
      kMean,
      kMeanDCC,
      kRMS,
      kRMSMap,
      kQualitySummary,
      kTrendMean,
      kTrendRMS,
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
    float toleranceMean_;
    float toleranceRMS_;
    float toleranceRMSFwd_;
    float noisyFracThreshold_;
  };

}

#endif

