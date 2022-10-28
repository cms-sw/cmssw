#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CondFormats/RunInfo/interface/LHCInfoPerLS.h"
#include "CondFormats/DataRecord/interface/LHCInfoPerLSRcd.h"

#include <memory>
#include <iostream>
#include <cassert>

class LHCInfoPerLSAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit LHCInfoPerLSAnalyzer(const edm::ParameterSet&)
      : tokenInfoPerLS_(esConsumes<LHCInfoPerLS, LHCInfoPerLSRcd>()) {}

private:
  void beginJob() override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override {}

  edm::ESWatcher<LHCInfoPerLSRcd> infoPerLSWatcher_;

  edm::ESGetToken<LHCInfoPerLS, LHCInfoPerLSRcd> tokenInfoPerLS_;
};

void LHCInfoPerLSAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get InfoPerLS
  assert(infoPerLSWatcher_.check(iSetup));
  const LHCInfoPerLS& infoPerLS = iSetup.getData(tokenInfoPerLS_);

  assert(infoPerLS.fillNumber() == 7066);
  assert(infoPerLS.lumiSection() == 1);
  assert(infoPerLS.crossingAngleX() == 170);
  assert(infoPerLS.crossingAngleY() == 170);
  assert(infoPerLS.betaStarX() == 11);
  assert(infoPerLS.betaStarY() == 11);
  assert(infoPerLS.runNumber() == 301765);
  edm::LogInfo("LHCInfoPerLSAnalyzer") << "LHCInfoPerLS retrieved:\n" << infoPerLS;
}

DEFINE_FWK_MODULE(LHCInfoPerLSAnalyzer);
