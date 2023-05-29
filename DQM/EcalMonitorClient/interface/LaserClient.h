#ifndef LaserClient_H
#define LaserClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {
  class LaserClient : public DQWorkerClient {
  public:
    LaserClient();
    ~LaserClient() override {}

    void producePlots(ProcessType) override;

  private:
    void setParams(edm::ParameterSet const&) override;

    std::map<int, unsigned> wlToME_;

    int minChannelEntries_;
    std::vector<float> expectedAmplitude_;
    float toleranceAmplitudeLo_;
    float toleranceAmplitudeFwdLo_;
    float toleranceAmplitudeHi_;
    float toleranceAmpRMSRatio_;
    std::vector<float> expectedTiming_;
    float toleranceTiming_;
    float toleranceTimRMS_;
    std::vector<float> expectedPNAmplitude_;
    float tolerancePNAmp_;
    float tolerancePNRMSRatio_;
    float forwardFactor_;
  };
}  // namespace ecaldqm

#endif
