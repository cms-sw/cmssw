#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1TCaloParamsRcd.h"
#include "CondFormats/DataRecord/interface/L1TCaloParamsO2ORcd.h"
#include "CondFormats/L1TObjects/interface/CaloParams.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

class L1TCaloStage2ParamsWriter : public edm::EDAnalyzer {
private:
  bool isO2Opayload;

public:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  explicit L1TCaloStage2ParamsWriter(const edm::ParameterSet& pset) : edm::EDAnalyzer() {
    isO2Opayload = pset.getUntrackedParameter<bool>("isO2Opayload", false);
  }
  ~L1TCaloStage2ParamsWriter(void) override {}
};

void L1TCaloStage2ParamsWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  edm::ESHandle<l1t::CaloParams> handle1;

  if (isO2Opayload)
    evSetup.get<L1TCaloParamsO2ORcd>().get(handle1);
  else
    evSetup.get<L1TCaloParamsRcd>().get(handle1);

  std::shared_ptr<l1t::CaloParams> ptr1(new l1t::CaloParams(*(handle1.product())));

  edm::Service<cond::service::PoolDBOutputService> poolDb;
  if (poolDb.isAvailable()) {
    cond::Time_t firstSinceTime = poolDb->beginOfTime();
    poolDb->writeOne(ptr1.get(), firstSinceTime, (isO2Opayload ? "L1TCaloParamsO2ORcd" : "L1TCaloParamsRcd"));
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1TCaloStage2ParamsWriter);
