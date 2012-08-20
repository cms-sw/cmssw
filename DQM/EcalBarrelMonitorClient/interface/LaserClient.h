#ifndef LaserClient_H
#define LaserClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class LaserClient : public DQWorkerClient {
  public:
    LaserClient(const edm::ParameterSet &);
    ~LaserClient() {}

    void bookMEs();

    void initialize();

    void beginRun(const edm::Run&, const edm::EventSetup&);

    void producePlots();

    enum Constants {
      nWL = 4,
      nPNGain = 2
    };

    enum MESets {
      kQuality,
      kAmplitudeMean = kQuality + nWL,
      kAmplitudeRMS = kAmplitudeMean + nWL,
      kTimingMean = kAmplitudeRMS + nWL,
      kTimingRMS = kTimingMean + nWL,
      kPNAmplitudeMean = kTimingRMS + nWL,
      kPNAmplitudeRMS = kPNAmplitudeMean + nWL * nPNGain,
      kQualitySummary = kPNAmplitudeRMS + nWL * nPNGain,
      kPNQualitySummary = kQualitySummary + nWL,
      nTargets = kPNQualitySummary + nWL,
      sAmplitude = 0,
      sTiming = sAmplitude + nWL,
      sPNAmplitude = sTiming + nWL,
      nSources = sPNAmplitude + nWL * nPNGain,
      nMESets = nTargets + nSources
    };

    static void setMEData(std::vector<MEData>&);

  protected:
    std::vector<int> laserWavelengths_;
    std::vector<int> MGPAGainsPN_;

    int minChannelEntries_;
    std::vector<double> expectedAmplitude_;
    std::vector<double> amplitudeThreshold_;
    std::vector<double> amplitudeRMSThreshold_;
    std::vector<double> expectedTiming_;
    std::vector<double> timingThreshold_;
    std::vector<double> timingRMSThreshold_;
    std::vector<double> expectedPNAmplitude_;
    std::vector<double> pnAmplitudeThreshold_;
    std::vector<double> pnAmplitudeRMSThreshold_;

    float towerThreshold_;

    std::map<std::pair<unsigned, int>, float> ampCorrections_;
  };

}

#endif
