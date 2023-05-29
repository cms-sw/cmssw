// -*- C++ -*-
//
// Package:    HGCalWaferTester
// Class:      HGCalWaferTester
//
/**\class HGCalWaferTester HGCalWaferTester.cc
 test/HGCalWaferTester.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2019/07/15
//
//

// system include files
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HGCalWaferTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalWaferTester(const edm::ParameterSet&);
  ~HGCalWaferTester() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const std::string nameSense_, nameDetector_;
  const bool reco_;
  edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> dddToken_;
};

HGCalWaferTester::HGCalWaferTester(const edm::ParameterSet& iC)
    : nameSense_(iC.getParameter<std::string>("NameSense")),
      nameDetector_(iC.getParameter<std::string>("NameDevice")),
      reco_(iC.getParameter<bool>("Reco")) {
  dddToken_ = esConsumes<HGCalDDDConstants, IdealGeometryRecord>(edm::ESInputTag{"", nameSense_});

  edm::LogVerbatim("HGCalGeom") << "Test numbering for " << nameDetector_ << " using constants of " << nameSense_
                                << " for  RecoFlag " << reco_;
}

void HGCalWaferTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("NameSense", "HGCalEESensitive");
  desc.add<std::string>("NameDevice", "HGCal EE");
  desc.add<bool>("Reco", false);
  descriptions.add("hgcalWaferTesterEE", desc);
}

// ------------ method called to produce the data  ------------
void HGCalWaferTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const HGCalDDDConstants& hgdc = iSetup.getData(dddToken_);
  edm::LogVerbatim("HGCalGeom") << nameDetector_ << " Layers = " << hgdc.layers(reco_)
                                << " Sectors = " << hgdc.sectors() << std::endl;
  if (hgdc.waferHexagon8()) {
    int layer = hgdc.firstLayer();
    for (int u = -12; u <= 12; ++u) {
      std::pair<double, double> xy = hgdc.waferPosition(layer, u, 0, reco_, false);
      edm::LogVerbatim("HGCalGeom") << " iz = +, u = " << u << ", v = 0: x = " << xy.first << " y = " << xy.second
                                    << "\n"
                                    << " iz = -, u = " << u << ", v = 0: x = " << -xy.first << " y = " << xy.second;
    }
    for (int v = -12; v <= 12; ++v) {
      std::pair<double, double> xy = hgdc.waferPosition(layer, 0, v, reco_, false);
      edm::LogVerbatim("HGCalGeom") << " iz = +, u = 0, v = " << v << ": x = " << xy.first << " y = " << xy.second
                                    << "\n"
                                    << " iz = -, u = 0, v = " << v << ": x = " << -xy.first << " y = " << xy.second;
    }
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalWaferTester);
