#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

//#include "CondFormats/DataRecord/interface/L1TMuonEndCapForestO2ORcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndCapForestRcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonEndCapForest.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

class L1TMuonEndCapForestWriter : public edm::EDAnalyzer {
private:
  bool isO2Opayload;

public:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  explicit L1TMuonEndCapForestWriter(const edm::ParameterSet& pset) : edm::EDAnalyzer() {
    isO2Opayload = pset.getUntrackedParameter<bool>("isO2Opayload", false);
  }
  ~L1TMuonEndCapForestWriter(void) override {}
};

void L1TMuonEndCapForestWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  edm::ESHandle<L1TMuonEndCapForest> handle1;

  //    if( isO2Opayload )
  //        evSetup.get<L1TMuonEndCapForestO2ORcd>().get( handle1 ) ;
  //    else
  evSetup.get<L1TMuonEndCapForestRcd>().get(handle1);

  std::shared_ptr<L1TMuonEndCapForest> ptr1(new L1TMuonEndCapForest(*(handle1.product())));

  edm::Service<cond::service::PoolDBOutputService> poolDb;
  if (poolDb.isAvailable()) {
    cond::Time_t firstSinceTime = poolDb->beginOfTime();
    poolDb->writeOne(
        ptr1.get(), firstSinceTime, (isO2Opayload ? "L1TMuonEndCapForestO2ORcd" : "L1TMuonEndCapForestRcd"));
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1TMuonEndCapForestWriter);
