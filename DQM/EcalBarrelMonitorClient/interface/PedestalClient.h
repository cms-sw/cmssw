#ifndef PedestalClient_H
#define PedestalClient_H

#include "DQWorkerClient.h"

namespace ecaldqm
{
  class PedestalClient : public DQWorkerClient {
  public:
    PedestalClient();
    ~PedestalClient() {}

    void producePlots(ProcessType) override;

  private:
    void setParams(edm::ParameterSet const&) override;

    std::map<int, unsigned> gainToME_;
    std::map<int, unsigned> pnGainToME_;

    int minChannelEntries_;
    float expectedMean_;
    float toleranceMean_;
    std::vector<float> toleranceRMS_;
    float expectedPNMean_;
    float tolerancePNMean_;
    std::vector<float> tolerancePNRMS_;
  };

}

#endif
