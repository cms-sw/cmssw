#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1TMuonGlobalParamsO2ORcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonGlobalParamsRcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonGlobalParams.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

class L1TMuonGlobalParamsWriter : public edm::one::EDAnalyzer<> {
private:
  edm::ESGetToken<L1TMuonGlobalParams, L1TMuonGlobalParamsO2ORcd> o2oToken_;
  edm::ESGetToken<L1TMuonGlobalParams, L1TMuonGlobalParamsRcd> token_;
  bool isO2Opayload;

public:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  explicit L1TMuonGlobalParamsWriter(const edm::ParameterSet& pset) {
    isO2Opayload = pset.getUntrackedParameter<bool>("isO2Opayload", false);
    if (isO2Opayload) {
      o2oToken_ = esConsumes();
    } else {
      token_ = esConsumes();
    }
  }
};

void L1TMuonGlobalParamsWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  edm::ESHandle<L1TMuonGlobalParams> handle1;

  if (isO2Opayload)
    handle1 = evSetup.getHandle(o2oToken_);
  else
    handle1 = evSetup.getHandle(token_);

  L1TMuonGlobalParams const& ptr1 = *handle1;

  edm::Service<cond::service::PoolDBOutputService> poolDb;
  if (poolDb.isAvailable()) {
    cond::Time_t firstSinceTime = poolDb->beginOfTime();
    poolDb->writeOneIOV(ptr1, firstSinceTime, (isO2Opayload ? "L1TMuonGlobalParamsO2ORcd" : "L1TMuonGlobalParamsRcd"));
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1TMuonGlobalParamsWriter);
