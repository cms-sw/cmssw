#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKeyExt.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyExtRcd.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

class L1KeyWriter : public edm::one::EDAnalyzer<> {
public:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  explicit L1KeyWriter(const edm::ParameterSet&) : token_{esConsumes()} {}

private:
  edm::ESGetToken<L1TriggerKeyExt, L1TriggerKeyExtRcd> token_;
};

void L1KeyWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  L1TriggerKeyExt const& ptr1 = evSetup.getData(token_);

  edm::Service<cond::service::PoolDBOutputService> poolDb;
  if (poolDb.isAvailable()) {
    cond::Time_t firstSinceTime = poolDb->beginOfTime();
    poolDb->writeOneIOV(ptr1, firstSinceTime, "L1TriggerKeyExtRcd");
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1KeyWriter);
