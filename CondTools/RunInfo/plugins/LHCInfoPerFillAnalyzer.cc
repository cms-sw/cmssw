#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CondFormats/RunInfo/interface/LHCInfoPerFill.h"
#include "CondFormats/DataRecord/interface/LHCInfoPerFillRcd.h"

#include <memory>
#include <iostream>
#include <vector>
#include <cassert>

class LHCInfoPerFillAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit LHCInfoPerFillAnalyzer(const edm::ParameterSet&)
      : tokenInfoPerFill_(esConsumes<LHCInfoPerFill, LHCInfoPerFillRcd>()) {}

private:
  void beginJob() override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override {}

  edm::ESWatcher<LHCInfoPerFillRcd> infoPerFillWatcher_;

  edm::ESGetToken<LHCInfoPerFill, LHCInfoPerFillRcd> tokenInfoPerFill_;
};

void LHCInfoPerFillAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  assert(infoPerFillWatcher_.check(iSetup));

  LHCInfoPerFill lhcInfoPerFill = iSetup.getData(tokenInfoPerFill_);
  const float EPS = 1E-4;
  assert(lhcInfoPerFill.fillNumber() == 3);
  assert(lhcInfoPerFill.bunchesInBeam1() == 10);
  assert(lhcInfoPerFill.bunchesInBeam2() == 8);
  assert(lhcInfoPerFill.collidingBunches() == 5);
  assert(lhcInfoPerFill.targetBunches() == 4);
  assert(lhcInfoPerFill.fillType() == lhcInfoPerFill.PROTONS);
  assert(lhcInfoPerFill.particleTypeForBeam1() == lhcInfoPerFill.PROTON);
  assert(lhcInfoPerFill.particleTypeForBeam2() == lhcInfoPerFill.PROTON);
  assert(abs(lhcInfoPerFill.intensityForBeam1() - 1016.5) < EPS);
  assert(abs(lhcInfoPerFill.intensityForBeam2() - 1096.66) < EPS);
  assert(abs(lhcInfoPerFill.energy() - 7000) < EPS);
  assert(abs(lhcInfoPerFill.delivLumi() - 2E-07) < EPS);
  assert(abs(lhcInfoPerFill.recLumi() - 2E-07) < EPS);
  assert(abs(lhcInfoPerFill.instLumi() - 0) < EPS);
  assert(abs(lhcInfoPerFill.instLumiError() - 0) < EPS);
  assert(lhcInfoPerFill.createTime() == 6561530930997627120);
  assert(lhcInfoPerFill.beginTime() == 6561530930997627120);
  assert(lhcInfoPerFill.endTime() == 6561530930997627120);
  assert(lhcInfoPerFill.injectionScheme() == "None");
  assert(lhcInfoPerFill.lumiPerBX().size() == 2);
  assert(abs(lhcInfoPerFill.lumiPerBX()[0] - 0.000114139) < EPS);
  assert(abs(lhcInfoPerFill.lumiPerBX()[1] - 0.000114139) < EPS);
  assert(lhcInfoPerFill.lhcState() == "some lhcState");
  assert(lhcInfoPerFill.lhcComment() == "some lhcComment");
  assert(lhcInfoPerFill.ctppsStatus() == "some ctppsStatus");
  edm::LogInfo("LHCInfoPerFillAnalyzer") << "LHCInfoPerFill retrieved:\n" << lhcInfoPerFill;
}

DEFINE_FWK_MODULE(LHCInfoPerFillAnalyzer);
