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

class testSiStripHistId : public edm::global::EDAnalyzer<> {
public:
  explicit testSiStripHistId(const edm::ParameterSet&);
  ~testSiStripHistId() = default;

private:
  void analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const& iSetup) const override;
};

testSiStripHistId::testSiStripHistId(const edm::ParameterSet& iConfig) {}

void testSiStripHistId::analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const& iSetup) const {
  // use SistripHistoId for producing histogram id (and title)
  SiStripHistoId hidmanager;
  edm::LogPrint("testSiStripHistId") << "------------------------------------";
  std::string hid1 = hidmanager.createHistoId("Cluster Distribution", "det", 2345698);
  edm::LogPrint("testSiStripHistId") << "created hid1: >>" << hid1 << "<<";
  std::string hid2 = hidmanager.createHistoId("Cluster Distribution", "fec", 1234);
  edm::LogPrint("testSiStripHistId") << "created hid2: >>" << hid2 << "<<";
  std::string hid3 = hidmanager.createHistoId("Cluster Distribution", "fed", 5678);
  edm::LogPrint("testSiStripHistId") << "created hid3: >>" << hid3 << "<<";
  edm::LogPrint("testSiStripHistId") << "------------------------------------";
  edm::LogPrint("testSiStripHistId") << "hid1 component id / component type: " << hidmanager.getComponentId(hid1)
                                     << " / " << hidmanager.getComponentType(hid1);
  edm::LogPrint("testSiStripHistId") << "hid2 component id / component type: " << hidmanager.getComponentId(hid2)
                                     << " / " << hidmanager.getComponentType(hid2);
  edm::LogPrint("testSiStripHistId") << "hid3 component id / component type: " << hidmanager.getComponentId(hid3)
                                     << " / " << hidmanager.getComponentType(hid3);
  edm::LogPrint("testSiStripHistId") << "------------------------------------";
  std::string hid4 = "just for_testing% _#31";
  edm::LogPrint("testSiStripHistId") << "hid4=" << hid4;
  edm::LogPrint("testSiStripHistId") << "hid4 component id / component type: " << hidmanager.getComponentId(hid4)
                                     << " / " << hidmanager.getComponentType(hid4);
  edm::LogPrint("testSiStripHistId") << "------------------------------------";
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(testSiStripHistId);
