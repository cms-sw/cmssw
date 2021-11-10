#include <iomanip>
#include <iostream>

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1TMuonOverlapFwVersionRcd.h"
#include "CondFormats/DataRecord/interface/L1TMuonOverlapFwVersionO2ORcd.h"
#include "CondFormats/L1TObjects/interface/L1TMuonOverlapFwVersion.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/CondDB/interface/Session.h"

using namespace std;

class L1TMuonOverlapFwVersionTester : public edm::EDAnalyzer {
private:
  bool isO2Opayload;
  bool writeToDB;

public:
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  explicit L1TMuonOverlapFwVersionTester(const edm::ParameterSet &pset) : edm::EDAnalyzer() {
    isO2Opayload = pset.getUntrackedParameter<bool>("isO2Opayload", false);
    writeToDB = pset.getUntrackedParameter<bool>("writeToDB", false);
  }
  ~L1TMuonOverlapFwVersionTester(void) override {}
};

void L1TMuonOverlapFwVersionTester::analyze(const edm::Event &iEvent, const edm::EventSetup &evSetup) {
  edm::ESHandle<L1TMuonOverlapFwVersion> handle1;
  if (isO2Opayload)
    evSetup.get<L1TMuonOverlapFwVersionO2ORcd>().get(handle1);
  else
    evSetup.get<L1TMuonOverlapFwVersionRcd>().get(handle1);
  std::shared_ptr<L1TMuonOverlapFwVersion> ptr1(new L1TMuonOverlapFwVersion(*(handle1.product())));

  cout << "Contents of L1TMuonOverlapFwVersion: " << endl;

  cout << " algoVersion() = " << ptr1->algoVersion() << endl;
  cout << " layersVersion() = " << ptr1->layersVersion() << endl;
  cout << " patternsVersion() = " << ptr1->patternsVersion() << endl;
  cout << " synthDate() = " << ptr1->synthDate() << endl;

  if (writeToDB) {
    edm::Service<cond::service::PoolDBOutputService> poolDb;
    if (poolDb.isAvailable()) {
      cout << "Writing payload to DB" << endl;
      cond::Time_t firstSinceTime = poolDb->beginOfTime();
      poolDb->writeOne(
          ptr1.get(), firstSinceTime, (isO2Opayload ? "L1TMuonOverlapFwVersionO2ORcd" : "L1TMuonOverlapFwVersionRcd"));
    }
  }
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ModuleFactory.h"

DEFINE_FWK_MODULE(L1TMuonOverlapFwVersionTester);
