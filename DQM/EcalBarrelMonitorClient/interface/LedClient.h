#ifndef LedClient_H
#define LedClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class LedClient : public DQWorkerClient {
  public:
    LedClient(edm::ParameterSet const&, edm::ParameterSet const&);
    ~LedClient() {}

    void beginRun(const edm::Run&, const edm::EventSetup&);

    void producePlots();

    enum MESets {
      kQuality,
      kAmplitudeMean,
      kAmplitudeRMS,
      kTimingMean,
      kTimingRMSMap,
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
    std::vector<double> expectedAmplitude_;
    std::vector<double> toleranceAmplitude_;
    std::vector<double> toleranceAmpRMSRatio_;
    std::vector<double> expectedTiming_;
    std::vector<double> toleranceTiming_;
    std::vector<double> toleranceTimRMS_;
    std::vector<double> expectedPNAmplitude_;
    std::vector<double> tolerancePNAmp_;
    std::vector<double> tolerancePNRMSRatio_;
    float forwardFactor_;
  };

}

#endif
