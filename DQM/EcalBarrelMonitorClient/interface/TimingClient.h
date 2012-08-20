#ifndef TimingClient_H
#define TimingClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class TimingClient : public DQWorkerClient {
  public:
    TimingClient(const edm::ParameterSet &);
    ~TimingClient() {}

    void bookMEs();

    void producePlots();

    enum MESets {
      kQuality,
      kMeanSM,
      kMeanAll,
      kFwdBkwdDiff,
      kFwdvBkwd,
      kRMS,
      kRMSAll,
      kProjEta,
      kProjPhi,
      kQualitySummary,
      nTargets,
      sTimeAllMap = 0,
      sTimeMap,
      nSources,
      nMESets = nTargets + nSources
    };

    static void setMEData(std::vector<MEData>&);

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

