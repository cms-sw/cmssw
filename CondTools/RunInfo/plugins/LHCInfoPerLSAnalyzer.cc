#include "CondFormats/Common/interface/Time.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "CondFormats/RunInfo/interface/LHCInfoPerLS.h"
#include "CondFormats/DataRecord/interface/LHCInfoPerLSRcd.h"

#include <memory>
#include <iostream>
#include <cassert>

class LHCInfoPerLSAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit LHCInfoPerLSAnalyzer(const edm::ParameterSet& pset)
      : tokenInfoPerLS_(esConsumes<LHCInfoPerLS, LHCInfoPerLSRcd>()),
        csvFormat_(pset.getUntrackedParameter<bool>("csvFormat")),
        header_(pset.getUntrackedParameter<bool>("header")),
        iov_(pset.getUntrackedParameter<cond::Time_t>("iov")),
        separator_(pset.getUntrackedParameter<std::string>("separator")) {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override {}

  edm::ESWatcher<LHCInfoPerLSRcd> infoPerLSWatcher_;

  const edm::ESGetToken<LHCInfoPerLS, LHCInfoPerLSRcd> tokenInfoPerLS_;
  const bool csvFormat_;
  const bool header_;
  const cond::Time_t iov_;
  const std::string separator_;
};

void LHCInfoPerLSAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<bool>("csvFormat", false);
  desc.addUntracked<bool>("header", false);
  desc.addUntracked<std::string>("separator", ",");
  desc.addUntracked<cond::Time_t>("iov", 0);
  descriptions.add("LHCInfoPerLSAnalyzer", desc);
}

void LHCInfoPerLSAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get InfoPerLS
  assert(infoPerLSWatcher_.check(iSetup));
  const LHCInfoPerLS& infoPerLS = iSetup.getData(tokenInfoPerLS_);

  if (csvFormat_) {
    auto s = separator_;
    if (header_) {
      std::cout << "IOV" << s << "class" << s << "timestamp" << s << "runNumber" << s << "LS" << s << "xangleX" << s
                << "xangleY" << s << "beta*X" << s << "beta*Y" << s << std::endl;
    }
    std::cout << iov_ << s << "LHCInfoPerLS" << s << iEvent.time().unixTime() << s << infoPerLS.runNumber() << s
              << infoPerLS.lumiSection() << s << infoPerLS.crossingAngleX() << s << infoPerLS.crossingAngleY() << s
              << infoPerLS.betaStarX() << s << infoPerLS.betaStarY() << s << std::endl;
  } else {
    std::cout << "LHCInfoPerLS retrieved:\n" << infoPerLS << std::endl;
  }
}

DEFINE_FWK_MODULE(LHCInfoPerLSAnalyzer);
