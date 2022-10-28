#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1TMuonEndCapParamsO2ORcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonEndCapParamsRcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonEndCapParams.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

class L1TMuonEndCapParamsWriter : public edm::one::EDAnalyzer<> {
private:
  edm::ESGetToken<L1TMuonEndCapParams, L1TMuonEndCapParamsO2ORcd> o2oToken_;
  edm::ESGetToken<L1TMuonEndCapParams, L1TMuonEndCapParamsRcd> token_;
  bool isO2Opayload;

public:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  explicit L1TMuonEndCapParamsWriter(const edm::ParameterSet& pset) {
    isO2Opayload = pset.getUntrackedParameter<bool>("isO2Opayload", false);
    if (isO2Opayload) {
      o2oToken_ = esConsumes();
    } else {
      token_ = esConsumes();
    }
  }
};

void L1TMuonEndCapParamsWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  edm::ESHandle<L1TMuonEndCapParams> handle1;

  if (isO2Opayload)
    handle1 = evSetup.getHandle(o2oToken_);
  else
    handle1 = evSetup.getHandle(token_);

  L1TMuonEndCapParams const& ptr1 = *handle1;

  edm::Service<cond::service::PoolDBOutputService> poolDb;
  if (poolDb.isAvailable()) {
    cond::Time_t firstSinceTime = poolDb->beginOfTime();
    poolDb->writeOneIOV(ptr1, firstSinceTime, (isO2Opayload ? "L1TMuonEndCapParamsO2ORcd" : "L1TMuonEndCapParamsRcd"));
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1TMuonEndCapParamsWriter);
