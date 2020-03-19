#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1TMuonBarrelParamsO2ORcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonBarrelParamsRcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonBarrelParams.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

class L1TMuonBarrelParamsWriter : public edm::EDAnalyzer {
private:
  bool isO2Opayload;

public:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  explicit L1TMuonBarrelParamsWriter(const edm::ParameterSet& pset) : edm::EDAnalyzer() {
    isO2Opayload = pset.getUntrackedParameter<bool>("isO2Opayload", false);
  }
  ~L1TMuonBarrelParamsWriter(void) override {}
};

void L1TMuonBarrelParamsWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  edm::ESHandle<L1TMuonBarrelParams> handle1;

  if (isO2Opayload)
    evSetup.get<L1TMuonBarrelParamsO2ORcd>().get(handle1);
  else
    evSetup.get<L1TMuonBarrelParamsRcd>().get(handle1);

  std::shared_ptr<L1TMuonBarrelParams> ptr1(new L1TMuonBarrelParams(*(handle1.product())));

  edm::Service<cond::service::PoolDBOutputService> poolDb;
  if (poolDb.isAvailable()) {
    cond::Time_t firstSinceTime = poolDb->beginOfTime();
    poolDb->writeOne(
        ptr1.get(), firstSinceTime, (isO2Opayload ? "L1TMuonBarrelParamsO2ORcd" : "L1TMuonBarrelParamsRcd"));
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1TMuonBarrelParamsWriter);
