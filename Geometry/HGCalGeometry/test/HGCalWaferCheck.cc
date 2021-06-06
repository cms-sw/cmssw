// -*- C++ -*-
//
// Package:    HGCalWaferCheck
// Class:      HGCalWaferCheck
//
/**\class HGCalWaferCheck HGCalWaferCheck.cc
 test/HGCalWaferCheck.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2019/07/17
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
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HGCalWaferCheck : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalWaferCheck(const edm::ParameterSet&);
  ~HGCalWaferCheck() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const std::string nameSense_, nameDetector_;
  const bool reco_;
  const edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> dddToken_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> geomToken_;
};

HGCalWaferCheck::HGCalWaferCheck(const edm::ParameterSet& iC)
    : nameSense_(iC.getParameter<std::string>("NameSense")),
      nameDetector_(iC.getParameter<std::string>("NameDevice")),
      reco_(iC.getParameter<bool>("Reco")),
      dddToken_(esConsumes<HGCalDDDConstants, IdealGeometryRecord>(edm::ESInputTag{"", nameSense_})),
      geomToken_(esConsumes<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag{"", nameSense_})) {
  std::cout << "Test numbering for " << nameDetector_ << " using constants of " << nameSense_ << " for  RecoFlag "
            << reco_ << std::endl;
}

HGCalWaferCheck::~HGCalWaferCheck() {}

// ------------ method called to produce the data  ------------
void HGCalWaferCheck::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const HGCalDDDConstants& hgdc = iSetup.getData(dddToken_);
  const auto& geomR = iSetup.getData(geomToken_);
  const HGCalGeometry* geom = &geomR;
  std::cout << nameDetector_ << " Layers = " << hgdc.layers(reco_) << " Sectors = " << hgdc.sectors() << std::endl;
  if (hgdc.waferHexagon8()) {
    DetId::Detector det = (nameSense_ == "HGCalHESiliconSensitive") ? DetId::HGCalHSi : DetId::HGCalEE;
    for (int layer = 1; layer <= 2; ++layer) {
      for (int waferU = -12; waferU <= 12; ++waferU) {
        int waferV(0);
        int type = hgdc.waferType(layer, waferU, waferV);
        int cell = (type == 0) ? 12 : 8;
        HGCSiliconDetId id1(det, 1, type, layer, waferU, waferV, cell, cell);
        if (geom->topology().valid(id1))
          std::cout << " ID: " << id1 << " Position " << geom->getPosition(id1) << std::endl;
        HGCSiliconDetId id2(det, -1, type, layer, waferU, waferV, cell, cell);
        if (geom->topology().valid(id2))
          std::cout << " ID: " << id2 << " Position " << geom->getPosition(id2) << std::endl;
      }
      for (int waferV = -12; waferV <= 12; ++waferV) {
        int waferU(0);
        int type = hgdc.waferType(layer, waferU, waferV);
        int cell = (type == 0) ? 12 : 8;
        HGCSiliconDetId id1(det, 1, type, layer, waferU, waferV, cell, cell);
        if (geom->topology().valid(id1))
          std::cout << " ID: " << id1 << " Position " << geom->getPosition(id1) << std::endl;
        HGCSiliconDetId id2(det, -1, type, layer, waferU, waferV, cell, cell);
        if (geom->topology().valid(id2))
          std::cout << " ID: " << id2 << " Position " << geom->getPosition(id2) << std::endl;
      }
    }
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalWaferCheck);
