#ifndef TimingClient_H
#define TimingClient_H

#include "DQM/EcalCommon/interface/DQWorkerClient.h"

namespace ecaldqm {

  class TimingClient : public DQWorkerClient {
  public:
    TimingClient(const edm::ParameterSet &, const edm::ParameterSet &);
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
      nMESets
    };

    static void setMEData(std::vector<MEData>&);

    enum Sources {
      sTimeAllMap,
      sTimeMap,
      nSources
    };

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

