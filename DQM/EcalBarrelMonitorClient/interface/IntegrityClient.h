#ifndef IntegrityClient_H
#define IntegrityClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class IntegrityClient : public DQWorkerClient {
  public:
    IntegrityClient(edm::ParameterSet const&, edm::ParameterSet const&);
    ~IntegrityClient() {}

    void producePlots();

    enum MESets {
      kQuality,
      kQualitySummary,
      nMESets
    };

    enum Sources {
      kOccupancy,
      kGain,
      kChId,
      kGainSwitch,
      kTowerId,
      kBlockSize,
      nSources
    };

    static void setMEOrdering(std::map<std::string, unsigned>&);

  protected:
    float errFractionThreshold_;
  };

}

#endif

