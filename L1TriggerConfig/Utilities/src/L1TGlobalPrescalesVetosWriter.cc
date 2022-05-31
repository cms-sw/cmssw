#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1TGlobalPrescalesVetosO2ORcd.h"
#include "CondFormats/DataRecord/interface/L1TGlobalPrescalesVetosRcd.h"
#include "CondFormats/L1TObjects/interface/L1TGlobalPrescalesVetos.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

class L1TGlobalPrescalesVetosWriter : public edm::one::EDAnalyzer<> {
private:
  edm::ESGetToken<L1TGlobalPrescalesVetos, L1TGlobalPrescalesVetosO2ORcd> o2oToken_;
  edm::ESGetToken<L1TGlobalPrescalesVetos, L1TGlobalPrescalesVetosRcd> token_;
  bool isO2Opayload;

public:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  explicit L1TGlobalPrescalesVetosWriter(const edm::ParameterSet& pset) {
    isO2Opayload = pset.getUntrackedParameter<bool>("isO2Opayload", false);
    if (isO2Opayload) {
      o2oToken_ = esConsumes();
    } else {
      token_ = esConsumes();
    }
  }
};

void L1TGlobalPrescalesVetosWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& evSetup) {
  edm::ESHandle<L1TGlobalPrescalesVetos> handle1;

  if (isO2Opayload)
    handle1 = evSetup.getHandle(o2oToken_);
  else
    handle1 = evSetup.getHandle(token_);

  L1TGlobalPrescalesVetos const& ptr1 = *handle1;

  edm::Service<cond::service::PoolDBOutputService> poolDb;
  if (poolDb.isAvailable()) {
    cond::Time_t firstSinceTime = poolDb->beginOfTime();
    poolDb->writeOneIOV(
        ptr1, firstSinceTime, (isO2Opayload ? "L1TGlobalPrescalesVetosO2ORcd" : "L1TGlobalPrescalesVetosRcd"));
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1TGlobalPrescalesVetosWriter);
