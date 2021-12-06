#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"

#include "CondFormats/DataRecord/interface/L1TMuonOverlapFwVersionRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonOverlapFwVersionO2ORcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonOverlapFwVersion.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/CondDB/interface/Session.h"

using namespace std;

class L1TMuonOverlapFwVersionTester : public edm::one::EDAnalyzer<> {
private:
  bool isO2Opayload;
  bool writeToDB;
  edm::ESGetToken<L1TMuonOverlapFwVersion, L1TMuonOverlapFwVersionRcd> esToken;
  edm::ESGetToken<L1TMuonOverlapFwVersion, L1TMuonOverlapFwVersionO2ORcd> esToken_O2O;

public:
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  explicit L1TMuonOverlapFwVersionTester(const edm::ParameterSet &pset) : edm::one::EDAnalyzer<>() {
    isO2Opayload = pset.getUntrackedParameter<bool>("isO2Opayload", false);
    writeToDB = pset.getUntrackedParameter<bool>("writeToDB", false);
    esToken = esConsumes<L1TMuonOverlapFwVersion, L1TMuonOverlapFwVersionRcd>();
    esToken_O2O = esConsumes<L1TMuonOverlapFwVersion, L1TMuonOverlapFwVersionO2ORcd>();
  }
  ~L1TMuonOverlapFwVersionTester(void) override = default;
};

void L1TMuonOverlapFwVersionTester::analyze(const edm::Event &iEvent, const edm::EventSetup &evSetup) {
  L1TMuonOverlapFwVersion data;
  if (isO2Opayload)
    data = evSetup.getData(esToken_O2O);
  else
    data = evSetup.getData(esToken);

  edm::LogInfo("L1TMuonOverlapFwVersionTester")
      << "*** Contents of L1TMuonOverlapFwVersion: algoVersion() = " << data.algoVersion()
      << ", layersVersion() = " << data.layersVersion() << ", patternsVersion() = " << data.patternsVersion()
      << ", synthDate() = " << data.synthDate();

  if (writeToDB) {
    edm::Service<cond::service::PoolDBOutputService> poolDb;
    if (poolDb.isAvailable()) {
      edm::LogInfo("L1TMuonOverlapFwVersionTester") << "*** Writing payload to DB";
      cond::Time_t firstSinceTime = poolDb->beginOfTime();
      poolDb->writeOneIOV(
          data, firstSinceTime, (isO2Opayload ? "L1TMuonOverlapFwVersionO2ORcd" : "L1TMuonOverlapFwVersionRcd"));
    }
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1TMuonOverlapFwVersionTester);
