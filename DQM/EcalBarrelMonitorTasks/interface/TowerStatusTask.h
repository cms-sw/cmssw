#ifndef TowerStatusTask_H
#define TowerStatusTask_H

#include "DQM/EcalCommon/interface/DQWorkerTask.h"

namespace ecaldqm {

  class TowerStatusTask : public DQWorkerTask {
  public:
    TowerStatusTask(const edm::ParameterSet &, const edm::ParameterSet &);
    ~TowerStatusTask();

    void bookMEs() override;

    void beginRun(const edm::Run &, const edm::EventSetup &) override;
    void endRun(const edm::Run &, const edm::EventSetup &) override;
    void beginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &) override;
    void endLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &) override;

    void runOnTowerStatus(const std::map<uint32_t, bool>&, int);

    enum MESets {
      kDAQSummary,
      kDAQSummaryMap,
      kDAQContents,
      kDCSSummary,
      kDCSSummaryMap,
      kDCSContents,
      nMESets
    };

    static void setMEData(std::vector<MEData>&);

  private:
    std::map<uint32_t, bool> daqLumiStatus_, daqRunStatus_;
    std::map<uint32_t, bool> dcsLumiStatus_, dcsRunStatus_;
    bool doDAQInfo_, doDCSInfo_;
  };

}

#endif

