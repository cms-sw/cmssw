// -*- C++ -*-
//
// Package:    HGCalValidHexTester
// Class:      HGCalValidHexTester
//
/**\class HGCalValidHexTester HGCalValidHexTester.cc
 test/HGCalValidHexTester.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2019/10/08
//
//

// system include files
#include <iostream>
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HGCalValidHexTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalValidHexTester(const edm::ParameterSet&);
  ~HGCalValidHexTester() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> dddToken_;
  std::string nameDetector_, nameSense_;
  std::vector<int> layers_, moduleU_, moduleV_, types_;
};

HGCalValidHexTester::HGCalValidHexTester(const edm::ParameterSet& iC) {
  nameDetector_ = iC.getParameter<std::string>("NameDevice");
  nameSense_ = iC.getParameter<std::string>("NameSense");
  layers_ = iC.getParameter<std::vector<int> >("Layers");
  moduleU_ = iC.getParameter<std::vector<int> >("ModuleU");
  moduleV_ = iC.getParameter<std::vector<int> >("ModuleV");
  types_ = iC.getParameter<std::vector<int> >("Types");

  dddToken_ = esConsumes<HGCalDDDConstants, IdealGeometryRecord>(edm::ESInputTag{"", nameSense_});

  std::cout << "Test valid cells for " << nameDetector_ << " using constants of " << nameSense_ << " for "
            << layers_.size() << " modules" << std::endl;
  for (unsigned int k = 0; k < layers_.size(); ++k)
    std::cout << "Wafer[" << k << "] Layer " << layers_[k] << " Type " << types_[k] << " U:V " << moduleU_[k] << ":"
              << moduleV_[k] << std::endl;
}

HGCalValidHexTester::~HGCalValidHexTester() {}

// ------------ method called to produce the data  ------------
void HGCalValidHexTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const DetId::Detector det = (nameSense_ == "HGCalEESensitive") ? DetId::HGCalEE : DetId::HGCalHSi;
  const HGCalDDDConstants& hgdc = iSetup.getData(dddToken_);
  std::cout << nameDetector_ << " Layers = " << hgdc.layers(true) << " Sectors = " << hgdc.sectors() << std::endl
            << std::endl;
  for (unsigned int k = 0; k < layers_.size(); ++k) {
    int nCells = (types_[k] == 0) ? HGCSiliconDetId::HGCalFineN : HGCSiliconDetId::HGCalCoarseN;
    int ncell(0);
    for (int u = 0; u < 2 * nCells; ++u) {
      for (int v = 0; v < 2 * nCells; ++v) {
        if (((v - u) < nCells) && (u - v) <= nCells) {
          std::string state = hgdc.isValidHex8(layers_[k], moduleU_[k], moduleV_[k], u, v) ? "within" : "outside of";
          std::cout << "Cell[" << k << "," << ncell << "] "
                    << HGCSiliconDetId(det, 1, types_[k], layers_[k], moduleU_[k], moduleV_[k], u, v) << " is " << state
                    << " fiducial volume" << std::endl;
          ++ncell;
        }
      }
    }
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalValidHexTester);
