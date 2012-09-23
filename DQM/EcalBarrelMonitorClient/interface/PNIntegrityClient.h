#ifndef PNIntegrityClient_H
#define PNIntegrityClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class PNIntegrityClient : public DQWorkerClient {
  public:
    PNIntegrityClient(edm::ParameterSet const&, edm::ParameterSet const&);
    ~PNIntegrityClient() {}

    void producePlots();

    enum MESets {
      kQualitySummary,
      nMESets
    };

    enum Sources {
      kOccupancy,
      kMEMChId,
      kMEMGain,
      kMEMBlockSize,
      kMEMTowerId,
      nSources
    };

    static void setMEOrdering(std::map<std::string, unsigned>&);

  protected:
    float errFractionThreshold_;
  };

}

#endif

