#ifndef LaserClient_H
#define LaserClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class LaserClient : public DQWorkerClient {
  public:
    LaserClient(edm::ParameterSet const&, edm::ParameterSet const&);
    ~LaserClient() {}

    void producePlots();

    enum MESets {
      kQuality,
      kAmplitudeMean,
      kAmplitudeRMS,
      kTimingMean,
      kTimingRMSMap,
      kTimingRMS,
      kQualitySummary,
      kPNQualitySummary,
      nMESets
    };

    enum Sources {
      kAmplitude,
      kTiming,
      kPNAmplitude,
      nSources
    };
 
    static void setMEOrdering(std::map<std::string, unsigned>&);

  protected:
    std::map<int, unsigned> wlToME_;

    int minChannelEntries_;
    std::vector<float> expectedAmplitude_;
    float toleranceAmplitude_;
    float toleranceAmpRMSRatio_;
    std::vector<float> expectedTiming_;
    float toleranceTiming_;
    float toleranceTimRMS_;
    std::vector<float> expectedPNAmplitude_;
    float tolerancePNAmp_;
    float tolerancePNRMSRatio_;
    float forwardFactor_;

  };

}

#endif
