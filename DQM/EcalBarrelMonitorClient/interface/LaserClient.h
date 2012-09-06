#ifndef LaserClient_H
#define LaserClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class LaserClient : public DQWorkerClient {
  public:
    LaserClient(edm::ParameterSet const&, edm::ParameterSet const&);
    ~LaserClient() {}

    void beginRun(const edm::Run&, const edm::EventSetup&);

    void producePlots();

    enum MESets {
      kQuality,
      kAmplitudeMean,
      kAmplitudeRMS,
      kTimingMean,
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
    std::vector<float> toleranceAmplitude_;
    std::vector<float> toleranceAmpRMSRatio_;
    std::vector<float> expectedTiming_;
    std::vector<float> toleranceTiming_;
    std::vector<float> toleranceTimRMS_;
    std::vector<float> expectedPNAmplitude_;
    std::vector<float> tolerancePNAmp_;
    std::vector<float> tolerancePNRMS_;
    float forwardFactor_;

  };

}

#endif
