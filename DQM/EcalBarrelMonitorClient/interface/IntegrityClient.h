#ifndef IntegrityClient_H
#define IntegrityClient_H

#include "DQM/EcalCommon/interface/DQWorkerClient.h"

namespace ecaldqm {

  class IntegrityClient : public DQWorkerClient {
  public:
    IntegrityClient(const edm::ParameterSet &, const edm::ParameterSet&);
    ~IntegrityClient() {}

    void bookMEs();

    void producePlots();

    enum MESets {
      kQuality,
      kQualitySummary,
      nMESets
    };

    static void setMEData(std::vector<MEData>&);

    enum Sources {
      sOccupancy,
      sGain,
      sChId,
      sGainSwitch,
      sTowerId,
      sBlockSize,
      nSources
    };

  protected:
    float errFractionThreshold_;
  };

}

#endif

