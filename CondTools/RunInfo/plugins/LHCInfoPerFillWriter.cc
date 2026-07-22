#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/RunInfo/interface/LHCInfoPerFill.h"
#include "CondFormats/DataRecord/interface/LHCInfoPerFillRcd.h"

#include <memory>
#include <iostream>
#include <vector>

class LHCInfoPerFillWriter : public edm::one::EDAnalyzer<> {
public:
  explicit LHCInfoPerFillWriter(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override {}

  size_t _sizeLumiPerBX;
};

LHCInfoPerFillWriter::LHCInfoPerFillWriter(const edm::ParameterSet& iConfig)
    : _sizeLumiPerBX(iConfig.getUntrackedParameter<int>("size", 0)) {}

void LHCInfoPerFillWriter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.addUntracked<int>("size", 0)
      ->setComment("Number of BX luminosity entries to generate");

  descriptions.add("LHCInfoPerFillWriter", desc);
}

void LHCInfoPerFillWriter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  LHCInfoPerFill lhcInfoPerFill;

  lhcInfoPerFill.setFillNumber(3);
  lhcInfoPerFill.setBunchesInBeam1(10);
  lhcInfoPerFill.setBunchesInBeam2(8);
  lhcInfoPerFill.setCollidingBunches(5);
  lhcInfoPerFill.setTargetBunches(4);
  lhcInfoPerFill.setFillType(lhcInfoPerFill.PROTONS);
  lhcInfoPerFill.setParticleTypeForBeam1(lhcInfoPerFill.PROTON);
  lhcInfoPerFill.setParticleTypeForBeam2(lhcInfoPerFill.PROTON);
  lhcInfoPerFill.setIntensityForBeam1(1016.5);
  lhcInfoPerFill.setIntensityForBeam2(1096.66);
  lhcInfoPerFill.setEnergy(7000);
  lhcInfoPerFill.setDelivLumi(2E-07);
  lhcInfoPerFill.setRecLumi(2E-07);
  lhcInfoPerFill.setInstLumi(0.);
  lhcInfoPerFill.setInstLumiError(0.);
  lhcInfoPerFill.setCreationTime(6561530930997627120);
  lhcInfoPerFill.setBeginTime(6561530930997627120);
  lhcInfoPerFill.setEndTime(6561530930997627120);
  lhcInfoPerFill.setInjectionScheme("None");

  // -------------------------------------
  // Create lumiPerBX with the given size
  // -------------------------------------
  std::vector<float> lumiPerBX(_sizeLumiPerBX, 0.0);
  // Fill with random values
  srand(static_cast<unsigned int>(time(nullptr)));
  for (size_t i = 0; i < _sizeLumiPerBX; ++i) {
    //random value from -1000000 to 1000000
    lumiPerBX[i] = static_cast<float>(-1000000. + static_cast<double>(rand()) / RAND_MAX * 2000000.);
  }
  lhcInfoPerFill.setLumiPerBX(lumiPerBX);

  lhcInfoPerFill.setLhcState("some lhcState");
  lhcInfoPerFill.setLhcComment("some lhcComment");
  lhcInfoPerFill.setCtppsStatus("some ctppsStatus");

  // store the info into a DB object
  edm::Service<cond::service::PoolDBOutputService> poolDbService;
  if (poolDbService.isAvailable())
    poolDbService->writeOneIOV(lhcInfoPerFill, poolDbService->currentTime(), "LHCInfoPerFillRcd");
  else
    throw cms::Exception("LHCInfoPerFillWriter")
        << "PoolDBOutputService required.";
}

DEFINE_FWK_MODULE(LHCInfoPerFillWriter);
