#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1TMuonBarrelParamsO2ORcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonBarrelParamsRcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonBarrelParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonBarrelKalmanParamsRcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonBarrelKalmanParams.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

class L1TMuonBarrelParamsWriter : public edm::one::EDAnalyzer<> {
private:
  edm::ESGetToken<L1TMuonBarrelParams, L1TMuonBarrelParamsO2ORcd> o2oParamsToken_;
  edm::ESGetToken<L1TMuonBarrelParams, L1TMuonBarrelParamsRcd> paramsToken_;
  edm::ESGetToken<L1TMuonBarrelKalmanParams, L1TMuonBarrelKalmanParamsRcd> kalmanToken_;
  bool isO2Opayload;

public:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  explicit L1TMuonBarrelParamsWriter(const edm::ParameterSet& pset) {
    isO2Opayload = pset.getUntrackedParameter<bool>("isO2Opayload", false);
    if (isO2Opayload) {
      o2oParamsToken_ = esConsumes();
    } else {
      paramsToken_ = esConsumes();
      kalmanToken_ = esConsumes();
    }
  }
};

void L1TMuonBarrelParamsWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  edm::ESHandle<L1TMuonBarrelParams> handle1;
  edm::ESHandle<L1TMuonBarrelKalmanParams> handle2;

  if (isO2Opayload)
    handle1 = evSetup.getHandle(o2oParamsToken_);
  else {
    handle1 = evSetup.getHandle(paramsToken_);
    handle2 = evSetup.getHandle(kalmanToken_);
  }

  edm::Service<cond::service::PoolDBOutputService> poolDb;
  if (poolDb.isAvailable()) {
    cond::Time_t firstSinceTime = poolDb->beginOfTime();
    poolDb->writeOneIOV(
        *handle1, firstSinceTime, (isO2Opayload ? "L1TMuonBarrelParamsO2ORcd" : "L1TMuonBarrelParamsRcd"));
    if (not isO2Opayload)
      poolDb->writeOneIOV(*handle2, firstSinceTime, ("L1TMuonBarrelKalmanParamsRcd"));
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1TMuonBarrelParamsWriter);
