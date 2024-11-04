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

  const LHCInfoPerFill& lhcInfoPerFill = iSetup.getData(tokenInfoPerFill_);

  std::cout << "LHCInfoPerFill retrieved:\n" << lhcInfoPerFill;
}

DEFINE_FWK_MODULE(LHCInfoPerFillAnalyzer);
