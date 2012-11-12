#ifndef TestPulseClient_H
#define TestPulseClient_H

#include "DQWorkerClient.h"

namespace ecaldqm
{
  class TestPulseClient : public DQWorkerClient {
  public:
    TestPulseClient(edm::ParameterSet const&, edm::ParameterSet const&);
    ~TestPulseClient() {}

    void producePlots();

  protected:
    std::map<int, unsigned> gainToME_;
    std::map<int, unsigned> pnGainToME_;

    int minChannelEntries_;
    std::vector<float> amplitudeThreshold_;
    std::vector<float> toleranceRMS_;
    std::vector<float> PNAmplitudeThreshold_;
    std::vector<float> tolerancePNRMS_;
  };

}

#endif
