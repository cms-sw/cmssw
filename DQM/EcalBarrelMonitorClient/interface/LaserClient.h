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
      kPNAmplitudeMean,
      kPNAmplitudeRMS,
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
    std::map<std::pair<int, int>, unsigned> wlGainToME_;

    int minChannelEntries_;
    std::vector<float> expectedAmplitude_;
    std::vector<float> amplitudeThreshold_;
    std::vector<float> amplitudeRMSThreshold_;
    std::vector<float> expectedTiming_;
    std::vector<float> timingThreshold_;
    std::vector<float> timingRMSThreshold_;
    std::vector<float> expectedPNAmplitude_;
    std::vector<float> pnAmplitudeThreshold_;
    std::vector<float> pnAmplitudeRMSThreshold_;

    float towerThreshold_;

    std::map<std::pair<unsigned, int>, float> ampCorrections_;
  };

}

#endif
