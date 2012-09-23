#ifndef PedestalClient_H
#define PedestalClient_H

#include "DQWorkerClient.h"

namespace ecaldqm
{
  class PedestalClient : public DQWorkerClient {
  public:
    PedestalClient(edm::ParameterSet const&, edm::ParameterSet const&);
    ~PedestalClient() {}

    void producePlots();

    enum MESets {
      kQuality,
      kMean,
      kRMS,
      kPNRMS,
      kQualitySummary,
      kPNQualitySummary,
      nMESets
    };

    enum Sources {
      kPedestal,
      kPNPedestal,
      nSources
    };

    static void setMEOrdering(std::map<std::string, unsigned>&);

  protected:
    std::map<int, unsigned> gainToME_;
    std::map<int, unsigned> pnGainToME_;

    std::vector<float> expectedMean_;
    std::vector<float> toleranceMean_;
    std::vector<float> toleranceRMS_;
    std::vector<float> expectedPNMean_;
    std::vector<float> tolerancePNMean_;
    std::vector<float> tolerancePNRMS_;
  };

}

#endif
