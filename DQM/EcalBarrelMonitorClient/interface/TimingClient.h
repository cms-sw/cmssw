#ifndef TimingClient_H
#define TimingClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class TimingClient : public DQWorkerClient {
  public:
    TimingClient(edm::ParameterSet const&, edm::ParameterSet const&);
    ~TimingClient() {}

    void bookMEs();

    void producePlots();

    enum MESets {
      kQuality,
      kMeanSM,
      kMeanAll,
      kFwdBkwdDiff,
      kFwdvBkwd,
      kRMSMap,
      kRMSAll,
      kProjEta,
      kProjPhi,
      kQualitySummary,
      nMESets
    };

    enum Sources {
      kTimeAllMap,
      kTimeMap,
      nSources
    };

    static void setMEOrdering(std::map<std::string, unsigned>&);

  protected:
    float expectedMean_;
    float meanThreshold_;
    float rmsThreshold_;
    int minChannelEntries_;
    int minTowerEntries_;
    float tailPopulThreshold_;
  };

}

#endif

