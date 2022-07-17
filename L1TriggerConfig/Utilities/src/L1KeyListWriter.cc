#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKeyListExt.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListExtRcd.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

class L1KeyListWriter : public edm::one::EDAnalyzer<> {
public:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  explicit L1KeyListWriter(const edm::ParameterSet&) : token_{esConsumes()} {}

private:
  edm::ESGetToken<L1TriggerKeyListExt, L1TriggerKeyListExtRcd> token_;
};

void L1KeyListWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  L1TriggerKeyListExt const& ptr1 = evSetup.getData(token_);

  edm::Service<cond::service::PoolDBOutputService> poolDb;
  if (poolDb.isAvailable()) {
    cond::Time_t firstSinceTime = poolDb->beginOfTime();
    poolDb->writeOneIOV(ptr1, firstSinceTime, "L1TriggerKeyListExtRcd");
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1KeyListWriter);
