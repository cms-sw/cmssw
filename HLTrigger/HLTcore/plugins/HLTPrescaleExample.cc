#include <string>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTPrescaleProvider.h"

class HLTPrescaleExample : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  HLTPrescaleExample(edm::ParameterSet const& iPSet);

  void beginJob() override {}
  void beginRun(edm::Run const& iEvent, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endRun(edm::Run const& iEvent, edm::EventSetup const&) override {}
  void endJob() override {}

private:
  HLTPrescaleProvider hltPSProvider_;
  std::string const hltProcess_;
  std::string const hltPath_;
};

HLTPrescaleExample::HLTPrescaleExample(edm::ParameterSet const& iPSet)
    : hltPSProvider_(iPSet.getParameter<edm::ParameterSet>("hltPSProvCfg"), consumesCollector(), *this),
      hltProcess_(iPSet.getParameter<std::string>("hltProcess")),
      hltPath_(iPSet.getParameter<std::string>("hltPath")) {}

void HLTPrescaleExample::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup) {
  bool changed = false;
  hltPSProvider_.init(iRun, iSetup, hltProcess_, changed);
}

void HLTPrescaleExample::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  auto const hltPSDouble = hltPSProvider_.prescaleValue<double>(iEvent, iSetup, hltPath_);
  auto const hltPSFrac = hltPSProvider_.prescaleValue<FractionalPrescale>(iEvent, iSetup, hltPath_);

  auto const l1HLTPSDouble = hltPSProvider_.prescaleValues<double>(iEvent, iSetup, hltPath_);
  auto const l1HLTPSFrac = hltPSProvider_.prescaleValues<FractionalPrescale>(iEvent, iSetup, hltPath_);
  auto const l1HLTPSDoubleFrac = hltPSProvider_.prescaleValues<double, FractionalPrescale>(iEvent, iSetup, hltPath_);

  auto const l1HLTDetailPSDouble = hltPSProvider_.prescaleValuesInDetail<double>(iEvent, iSetup, hltPath_);
  auto const l1HLTDetailPSFrac = hltPSProvider_.prescaleValuesInDetail<FractionalPrescale>(iEvent, iSetup, hltPath_);

  edm::LogPrint log("");

  log << "---------Begin Event--------\n";
  log << "hltDouble " << hltPSDouble << " hltFrac " << hltPSFrac << "\n";
  log << " l1HLTDouble " << l1HLTPSDouble.first << " " << l1HLTPSDouble.second << " l1HLTFrac " << l1HLTPSFrac.first
      << " " << l1HLTPSFrac.second << " l1HLTDoubleFrac " << l1HLTPSDoubleFrac.first << " " << l1HLTPSDoubleFrac.second
      << "\n";
  auto printL1HLTDetail = [&log](const std::string& text, const auto& val) {
    log << text;
    for (const auto& entry : val.first) {
      log << entry.first << ":" << entry.second << " ";
    }
    log << " HLT : " << val.second << "\n";
  };

  printL1HLTDetail("l1HLTDetailDouble ", l1HLTDetailPSDouble);
  printL1HLTDetail("l1HLTDetailFrac ", l1HLTDetailPSFrac);
  log << "---------End Event--------\n\n";
}

DEFINE_FWK_MODULE(HLTPrescaleExample);
