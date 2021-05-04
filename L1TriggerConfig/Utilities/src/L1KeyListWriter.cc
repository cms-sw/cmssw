#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKeyListExt.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListExtRcd.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

class L1KeyListWriter : public edm::EDAnalyzer {
public:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  explicit L1KeyListWriter(const edm::ParameterSet&) : edm::EDAnalyzer() {}
  ~L1KeyListWriter(void) override {}
};

void L1KeyListWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  edm::ESHandle<L1TriggerKeyListExt> handle1;
  evSetup.get<L1TriggerKeyListExtRcd>().get(handle1);
  std::shared_ptr<L1TriggerKeyListExt> ptr1(new L1TriggerKeyListExt(*(handle1.product())));

  edm::Service<cond::service::PoolDBOutputService> poolDb;
  if (poolDb.isAvailable()) {
    cond::Time_t firstSinceTime = poolDb->beginOfTime();
    poolDb->writeOne(ptr1.get(), firstSinceTime, "L1TriggerKeyListExtRcd");
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1KeyListWriter);
