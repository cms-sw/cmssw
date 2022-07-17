
//
// Original Author:  Dorian Kcira
//         Created:  Thu Feb 23 18:50:29 CET 2006
//
//
// test HistId classes of SiStrip

// system include files
#include <memory>
#include <iostream>

// user include files
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class Test_SiStrip_HistId : public edm::global::EDAnalyzer<> {
public:
  explicit Test_SiStrip_HistId(const edm::ParameterSet&);
  ~Test_SiStrip_HistId() = default;

private:
  void analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const& iSetup) const override;
};

Test_SiStrip_HistId::Test_SiStrip_HistId(const edm::ParameterSet& iConfig) {}

void Test_SiStrip_HistId::analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const& iSetup) const {
  // use SistripHistoId for producing histogram id (and title)
  SiStripHistoId hidmanager;
  edm::LogPrint("Test_SiStrip_HistId") << "------------------------------------" << std::endl;
  std::string hid1 = hidmanager.createHistoId("Cluster Distribution", "det", 2345698);
  edm::LogPrint("Test_SiStrip_HistId") << "created hid1: >>" << hid1 << "<<" << std::endl;
  std::string hid2 = hidmanager.createHistoId("Cluster Distribution", "fec", 1234);
  edm::LogPrint("Test_SiStrip_HistId") << "created hid2: >>" << hid2 << "<<" << std::endl;
  std::string hid3 = hidmanager.createHistoId("Cluster Distribution", "fed", 5678);
  edm::LogPrint("Test_SiStrip_HistId") << "created hid3: >>" << hid3 << "<<" << std::endl;
  edm::LogPrint("Test_SiStrip_HistId") << "------------------------------------" << std::endl;
  edm::LogPrint("Test_SiStrip_HistId") << "hid1 component id / component type: " << hidmanager.getComponentId(hid1)
                                       << " / " << hidmanager.getComponentType(hid1) << std::endl;
  edm::LogPrint("Test_SiStrip_HistId") << "hid2 component id / component type: " << hidmanager.getComponentId(hid2)
                                       << " / " << hidmanager.getComponentType(hid2) << std::endl;
  edm::LogPrint("Test_SiStrip_HistId") << "hid3 component id / component type: " << hidmanager.getComponentId(hid3)
                                       << " / " << hidmanager.getComponentType(hid3) << std::endl;
  edm::LogPrint("Test_SiStrip_HistId") << "------------------------------------" << std::endl;
  std::string hid4 = "just for_testing% _#31";
  edm::LogPrint("Test_SiStrip_HistId") << "hid4=" << hid4 << std::endl;
  edm::LogPrint("Test_SiStrip_HistId") << "hid4 component id / component type: " << hidmanager.getComponentId(hid4)
                                       << " / " << hidmanager.getComponentType(hid4) << std::endl;
  edm::LogPrint("Test_SiStrip_HistId") << "------------------------------------" << std::endl;
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(Test_SiStrip_HistId);
