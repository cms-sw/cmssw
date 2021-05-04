#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsRcd.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

class L1TMuonOverlapParamsWriter : public edm::EDAnalyzer {
public:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  explicit L1TMuonOverlapParamsWriter(const edm::ParameterSet&) : edm::EDAnalyzer() {}
  ~L1TMuonOverlapParamsWriter(void) override {}
};

void L1TMuonOverlapParamsWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  edm::ESHandle<L1TMuonOverlapParams> handle1;
  evSetup.get<L1TMuonOverlapParamsRcd>().get(handle1);
  std::shared_ptr<L1TMuonOverlapParams> ptr1(new L1TMuonOverlapParams(*(handle1.product())));

  edm::Service<cond::service::PoolDBOutputService> poolDb;
  if (poolDb.isAvailable()) {
    cond::Time_t firstSinceTime = poolDb->beginOfTime();
    poolDb->writeOne(ptr1.get(), firstSinceTime, "L1TMuonOverlapPatternParamsRcd");
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1TMuonOverlapParamsWriter);
