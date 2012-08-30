#ifndef CalibrationSummaryClient_H
#define CalibrationSummaryClient_H

#include "DQWorkerClient.h"

namespace ecaldqm {

  class CalibrationSummaryClient : public DQWorkerClient {
  public:
    CalibrationSummaryClient(edm::ParameterSet const&, edm::ParameterSet const&);
    ~CalibrationSummaryClient() {}

    void beginRun(const edm::Run &, const edm::EventSetup &);

    void producePlots();

    enum MESets {
      kQualitySummary,
      kPNQualitySummary,
      nMESets
    };

    enum Sources {
      kPNIntegrity,
      kLaser,
      kLaserPN,
      kLed,
      kLedPN,
      kTestPulse,
      kTestPulsePN,
      kPedestal,
      kPedestalPN,
      nSources
    };

    static void setMEOrdering(std::map<std::string, unsigned>&);

  protected:
    std::map<int, unsigned> laserWlToME_;
    std::map<int, unsigned> ledWlToME_;
    std::map<int, unsigned> tpGainToME_;
    std::map<int, unsigned> tpPNGainToME_;
    std::map<int, unsigned> pedGainToME_;
    std::map<int, unsigned> pedPNGainToME_;
  };

}

#endif

