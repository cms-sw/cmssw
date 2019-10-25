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

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HGCalWaferTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalWaferTester(const edm::ParameterSet&);
  ~HGCalWaferTester() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> dddToken_;
  std::string nameSense_, nameDetector_;
  bool reco_;
};

HGCalWaferTester::HGCalWaferTester(const edm::ParameterSet& iC) {
  nameSense_ = iC.getParameter<std::string>("NameSense");
  nameDetector_ = iC.getParameter<std::string>("NameDevice");
  reco_ = iC.getParameter<bool>("Reco");

  dddToken_ = esConsumes<HGCalDDDConstants, IdealGeometryRecord>(edm::ESInputTag{"", nameSense_});

  std::cout << "Test numbering for " << nameDetector_ << " using constants of " << nameSense_ << " for  RecoFlag "
            << reco_ << std::endl;
}

HGCalWaferTester::~HGCalWaferTester() {}

// ------------ method called to produce the data  ------------
void HGCalWaferTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const HGCalDDDConstants& hgdc = iSetup.getData(dddToken_);
  std::cout << nameDetector_ << " Layers = " << hgdc.layers(reco_) << " Sectors = " << hgdc.sectors() << std::endl;
  if ((hgdc.geomMode() == HGCalGeometryMode::Hexagon8) || (hgdc.geomMode() == HGCalGeometryMode::Hexagon8Full)) {
    int layer = hgdc.firstLayer();
    for (int u = -12; u <= 12; ++u) {
      std::pair<double, double> xy = hgdc.waferPosition(layer, u, 0, reco_);
      std::cout << " iz = +, u = " << u << ", v = 0: x = " << xy.first << " y = " << xy.second << "\n"
                << " iz = -, u = " << u << ", v = 0: x = " << -xy.first << " y = " << xy.second << "\n";
    }
    for (int v = -12; v <= 12; ++v) {
      std::pair<double, double> xy = hgdc.waferPosition(layer, 0, v, reco_);
      std::cout << " iz = +, u = 0, v = " << v << ": x = " << xy.first << " y = " << xy.second << "\n"
                << " iz = -, u = 0, v = " << v << ": x = " << -xy.first << " y = " << xy.second << "\n";
    }
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalWaferTester);
