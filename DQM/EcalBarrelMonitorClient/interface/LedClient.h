#ifndef LedClient_H
#define LedClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class LedClient : public DQWorkerClient {
  public:
    LedClient(edm::ParameterSet const&, edm::ParameterSet const&);
    ~LedClient() {}

    void producePlots();

  protected:
    std::map<int, unsigned> wlToME_;

    int minChannelEntries_;
    std::vector<double> expectedAmplitude_;
    double toleranceAmplitude_;
    double toleranceAmpRMSRatio_;
    std::vector<double> expectedTiming_;
    double toleranceTiming_;
    double toleranceTimRMS_;
    std::vector<double> expectedPNAmplitude_;
    double tolerancePNAmp_;
    double tolerancePNRMSRatio_;
    float forwardFactor_;
  };

}

#endif
