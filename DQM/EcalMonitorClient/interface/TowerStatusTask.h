#ifndef TowerStatusTask_H
#define TowerStatusTask_H

#include "DQWorkerClient.h"

#include "DQM/EcalCommon/interface/EcalDQMCommonUtils.h"
#include "CondFormats/EcalObjects/interface/EcalDAQTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalDAQTowerStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalDCSTowerStatus.h"
#include "CondFormats/DataRecord/interface/EcalDCSTowerStatusRcd.h"

namespace ecaldqm {

  class TowerStatusTask : public DQWorkerClient {
  public:
    TowerStatusTask();
    ~TowerStatusTask() override {}

    void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

    void producePlots(ProcessType) override;

  private:
    void setParams(edm::ParameterSet const&) override;
    void producePlotsTask_(float const*, std::string const&);
    edm::ESGetToken<EcalDAQTowerStatus, EcalDAQTowerStatusRcd> daqHndlToken;
    edm::ESGetToken<EcalDCSTowerStatus, EcalDCSTowerStatusRcd> dcsHndlToken;
    void setTokens(edm::ConsumesCollector&) override;

    bool doDAQInfo_;
    bool doDCSInfo_;
    float daqStatus_[nDCC];
    float dcsStatus_[nDCC];
  };

}  // namespace ecaldqm

#endif
