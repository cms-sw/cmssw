#include "CondFormats/Common/interface/Time.h"
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
  explicit LHCInfoPerFillAnalyzer(const edm::ParameterSet& pset)
      : tokenInfoPerFill_(esConsumes<LHCInfoPerFill, LHCInfoPerFillRcd>()),
        iov_(pset.getUntrackedParameter<cond::Time_t>("iov")) {}

private:
  void beginJob() override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override {}

  edm::ESWatcher<LHCInfoPerFillRcd> infoPerFillWatcher_;

  edm::ESGetToken<LHCInfoPerFill, LHCInfoPerFillRcd> tokenInfoPerFill_;
  const cond::Time_t iov_;
};

void LHCInfoPerFillAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  assert(infoPerFillWatcher_.check(iSetup));

  const LHCInfoPerFill& lhcInfoPerFill = iSetup.getData(tokenInfoPerFill_);

  std::cout << "IOV " << iov_ << "\nLHCInfoPerFill retrieved:\n" << lhcInfoPerFill;
}

DEFINE_FWK_MODULE(LHCInfoPerFillAnalyzer);
