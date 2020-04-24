#ifndef CalibrationSummaryClient_H
#define CalibrationSummaryClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class CalibrationSummaryClient : public DQWorkerClient {
  public:
    CalibrationSummaryClient();
    ~CalibrationSummaryClient() {}

    void producePlots(ProcessType) override;

  private:
    void setParams(edm::ParameterSet const&) override;

    std::map<int, unsigned> laserWlToME_;
    std::map<int, unsigned> ledWlToME_;
    std::map<int, unsigned> tpGainToME_;
    std::map<int, unsigned> tpPNGainToME_;
    std::map<int, unsigned> pedGainToME_;
    std::map<int, unsigned> pedPNGainToME_;
  };

}

#endif

