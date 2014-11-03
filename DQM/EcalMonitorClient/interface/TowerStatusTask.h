#ifndef TowerStatusTask_H
#define TowerStatusTask_H

#include "DQWorkerClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"

namespace ecaldqm {

  class TowerStatusTask : public DQWorkerClient {
  public:
    TowerStatusTask();
    ~TowerStatusTask() {}

    void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

    void producePlots(ProcessType) override;

  private:
    void setParams(edm::ParameterSet const&) override;
    void producePlotsTask_(float const*, std::string const&);

    bool doDAQInfo_;
    bool doDCSInfo_;
    float daqStatus_[nDCC];
    float dcsStatus_[nDCC];
  };

}

#endif

