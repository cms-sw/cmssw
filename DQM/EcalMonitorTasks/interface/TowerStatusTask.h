#ifndef TowerStatusTask_H
#define TowerStatusTask_H

#include "DQWorkerTask.h"

namespace ecaldqm {

  class TowerStatusTask : public DQWorkerTask {
  public:
    enum InfoType {
      DAQInfo,
      DCSInfo
    };

    TowerStatusTask();
    ~TowerStatusTask() {}

    void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

    void runOnTowerStatus(float const*, InfoType);

  private:
    void setParams(edm::ParameterSet const&) override;

    bool doDAQInfo_;
    bool doDCSInfo_;
  };

}

#endif

