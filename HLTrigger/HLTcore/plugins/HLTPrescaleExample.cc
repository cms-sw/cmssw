
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTPrescaleProvider.h"
#include <iostream>

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
  std::string hltProcess_;
  std::string hltPath_;
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
  auto hltPSDouble = hltPSProvider_.prescaleValue<double>(iEvent, iSetup, hltPath_);
  auto hltPSInt = hltPSProvider_.prescaleValue<int>(iEvent, iSetup, hltPath_);
  auto hltPSUInt = hltPSProvider_.prescaleValue<unsigned int>(iEvent, iSetup, hltPath_);
  auto hltPSFrac = hltPSProvider_.prescaleValue<FractionalPrescale>(iEvent, iSetup, hltPath_);

  auto l1HLTPSDouble = hltPSProvider_.prescaleValues<double>(iEvent, iSetup, hltPath_);
  auto l1HLTPSInt = hltPSProvider_.prescaleValues<int>(iEvent, iSetup, hltPath_);
  auto l1HLTPSFrac = hltPSProvider_.prescaleValues<FractionalPrescale>(iEvent, iSetup, hltPath_);
  auto l1HLTPSDoubleFrac = hltPSProvider_.prescaleValues<double, FractionalPrescale>(iEvent, iSetup, hltPath_);

  auto l1HLTDetailPSDouble = hltPSProvider_.prescaleValuesInDetail<double>(iEvent, iSetup, hltPath_);
  auto l1HLTDetailPSInt = hltPSProvider_.prescaleValuesInDetail<int>(iEvent, iSetup, hltPath_);
  auto l1HLTDetailPSFrac = hltPSProvider_.prescaleValuesInDetail<FractionalPrescale>(iEvent, iSetup, hltPath_);

  std::cout << "---------Begin Event--------" << std::endl;
  std::cout << "hltDouble " << hltPSDouble << " hltInt " << hltPSInt << " hltPSUInt " << hltPSUInt << " hltFrac "
            << hltPSFrac << std::endl;

  std::cout << " l1HLTDouble " << l1HLTPSDouble.first << " " << l1HLTPSDouble.second << " l1HLTInt " << l1HLTPSInt.first
            << " " << l1HLTPSInt.second << " l1HLTFrac " << l1HLTPSFrac.first << " " << l1HLTPSFrac.second
            << " l1HLTDoubleFrac " << l1HLTPSDoubleFrac.first << " " << l1HLTPSDoubleFrac.second << std::endl;
  auto printL1HLTDetail = [](const std::string& text, const auto& val, std::ostream& out) {
    out << text;
    for (const auto& entry : val.first) {
      out << entry.first << ":" << entry.second << " ";
    }
    out << " HLT : " << val.second << std::endl;
  };

  printL1HLTDetail("l1HLTDetailDouble ", l1HLTDetailPSDouble, std::cout);
  printL1HLTDetail("l1HLTDetailInt ", l1HLTDetailPSInt, std::cout);
  printL1HLTDetail("l1HLTDetailFrac ", l1HLTDetailPSFrac, std::cout);
  std::cout << "---------End Event--------" << std::endl << std::endl;
}

DEFINE_FWK_MODULE(HLTPrescaleExample);
