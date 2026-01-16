#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "CondFormats/RunInfo/interface/LHCInfo.h"
#include "CondFormats/DataRecord/interface/LHCInfoRcd.h"

#include <memory>
#include <iostream>

class LHCInfoAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit LHCInfoAnalyzer(const edm::ParameterSet&) : tokenInfo_(esConsumes<LHCInfo, LHCInfoRcd>()) {}

private:
  void beginJob() override {}
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override {}

  edm::ESWatcher<LHCInfoRcd> InfoWatcher_;

  edm::ESGetToken<LHCInfo, LHCInfoRcd> tokenInfo_;
};

void LHCInfoAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get Info
  if (InfoWatcher_.check(iSetup)) {
    const auto& lhcInfo = iSetup.getData(tokenInfo_);
    std::cout << "LHCInfo;" << iEvent.time().unixTime() << ";" << lhcInfo.lumiSection() << ";"
              << lhcInfo.crossingAngle() << ";" << lhcInfo.betaStar() << ";" << lhcInfo.delivLumi() << ";" << std::endl;
  }
}

DEFINE_FWK_MODULE(LHCInfoAnalyzer);