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

    TowerStatusTask(edm::ParameterSet const&, edm::ParameterSet const&);

    void beginRun(const edm::Run &, const edm::EventSetup &);
    void beginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup &);

    void runOnTowerStatus(std::vector<float> const&, InfoType);

  private:
    bool doDAQInfo_;
    bool doDCSInfo_;
  };

}

#endif

