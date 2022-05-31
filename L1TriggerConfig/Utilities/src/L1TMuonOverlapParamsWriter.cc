#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/L1TObjects/interface/L1TMuonOverlapParams.h"
#include "CondFormats/DataRecord/interface/L1TMuonOverlapParamsRcd.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

class L1TMuonOverlapParamsWriter : public edm::one::EDAnalyzer<> {
public:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  explicit L1TMuonOverlapParamsWriter(const edm::ParameterSet&) : token_{esConsumes()} {}

private:
  edm::ESGetToken<L1TMuonOverlapParams, L1TMuonOverlapParamsRcd> token_;
};

void L1TMuonOverlapParamsWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  L1TMuonOverlapParams const& ptr1 = evSetup.getData(token_);

  edm::Service<cond::service::PoolDBOutputService> poolDb;
  if (poolDb.isAvailable()) {
    cond::Time_t firstSinceTime = poolDb->beginOfTime();
    poolDb->writeOneIOV(ptr1, firstSinceTime, "L1TMuonOverlapPatternParamsRcd");
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1TMuonOverlapParamsWriter);
