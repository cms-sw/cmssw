#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1TUtmTriggerMenuRcd.h"
#include "CondFormats/DataRecord/interface/L1TUtmTriggerMenuO2ORcd.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

class L1MenuWriter : public edm::one::EDAnalyzer<> {
private:
  bool isO2Opayload;

public:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  const edm::ESGetToken<L1TUtmTriggerMenu, L1TUtmTriggerMenuRcd> l1GtMenuToken_;
  const edm::ESGetToken<L1TUtmTriggerMenu, L1TUtmTriggerMenuO2ORcd> l1GtMenuO2OToken_;

  explicit L1MenuWriter(const edm::ParameterSet& pset)
      : edm::one::EDAnalyzer<>(),
        l1GtMenuToken_(esConsumes<L1TUtmTriggerMenu, L1TUtmTriggerMenuRcd>()),
        l1GtMenuO2OToken_(esConsumes<L1TUtmTriggerMenu, L1TUtmTriggerMenuO2ORcd>()) {
    isO2Opayload = pset.getUntrackedParameter<bool>("isO2Opayload", false);
  }
  ~L1MenuWriter(void) override = default;
};

void L1MenuWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  edm::ESHandle<L1TUtmTriggerMenu> handle1;

  if (isO2Opayload)
    handle1 = evSetup.getHandle(l1GtMenuO2OToken_);
  else
    handle1 = evSetup.getHandle(l1GtMenuToken_);

  std::shared_ptr<L1TUtmTriggerMenu> ptr1(new L1TUtmTriggerMenu(*(handle1.product())));

  edm::Service<cond::service::PoolDBOutputService> poolDb;
  if (poolDb.isAvailable()) {
    cond::Time_t firstSinceTime = poolDb->beginOfTime();
    poolDb->writeOneIOV(*ptr1, firstSinceTime, (isO2Opayload ? "L1TUtmTriggerMenuO2ORcd" : "L1TUtmTriggerMenuRcd"));
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1MenuWriter);
