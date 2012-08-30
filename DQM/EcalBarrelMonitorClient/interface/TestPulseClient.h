#ifndef TestPulseClient_H
#define TestPulseClient_H

#include "DQWorkerClient.h"

namespace ecaldqm
{
  class TestPulseClient : public DQWorkerClient {
  public:
    TestPulseClient(edm::ParameterSet const&, edm::ParameterSet const&);
    ~TestPulseClient() {}

    void beginRun(const edm::Run&, const edm::EventSetup&);

    void producePlots();

    enum MESets {
      kQuality,
      kAmplitudeRMS,
      kPNAmplitudeRMS,
      kQualitySummary,
      kPNQualitySummary,
      nMESets
    };

    enum Sources {
      kAmplitude,
      kPNAmplitude,
      nSources
    };

    static void setMEOrdering(std::map<std::string, unsigned>&);

  protected:
    std::map<int, unsigned> gainToME_;
    std::map<int, unsigned> pnGainToME_;

    std::vector<float> amplitudeThreshold_;
    std::vector<float> toleranceRMS_;
    std::vector<float> PNAmplitudeThreshold_;
    std::vector<float> tolerancePNRMS_;
  };

}

#endif
