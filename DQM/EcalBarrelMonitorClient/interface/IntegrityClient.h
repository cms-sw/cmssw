#ifndef IntegrityClient_H
#define IntegrityClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class IntegrityClient : public DQWorkerClient {
  public:
    IntegrityClient(const edm::ParameterSet &);
    ~IntegrityClient() {}

    void bookMEs();

    void producePlots();

    enum MESets {
      kQuality,
      kQualitySummary,
      nTargets,
      sOccupancy = 0,
      sGain,
      sChId,
      sGainSwitch,
      sTowerId,
      sBlockSize,
      nSources,
      nMESets = nTargets + nSources
    };

    static void setMEData(std::vector<MEData>&);

  protected:
    float errFractionThreshold_;
  };

}

#endif

