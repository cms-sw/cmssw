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
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HGCalValidHexTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalValidHexTester(const edm::ParameterSet&);
  ~HGCalValidHexTester() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const std::string nameDetector_, nameSense_;
  const std::vector<int> layers_, moduleU_, moduleV_, types_;
  edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> dddToken_;
};

HGCalValidHexTester::HGCalValidHexTester(const edm::ParameterSet& iC)
    : nameDetector_(iC.getParameter<std::string>("NameDevice")),
      nameSense_(iC.getParameter<std::string>("NameSense")),
      layers_(iC.getParameter<std::vector<int> >("Layers")),
      moduleU_(iC.getParameter<std::vector<int> >("ModuleU")),
      moduleV_(iC.getParameter<std::vector<int> >("ModuleV")),
      types_(iC.getParameter<std::vector<int> >("Types")) {
  dddToken_ = esConsumes<HGCalDDDConstants, IdealGeometryRecord>(edm::ESInputTag{"", nameSense_});

  edm::LogVerbatim("HGCalGeom") << "Test valid cells for " << nameDetector_ << " using constants of " << nameSense_
                                << " for " << layers_.size() << " modules";
  for (unsigned int k = 0; k < layers_.size(); ++k)
    edm::LogVerbatim("HGCalGeom") << "Wafer[" << k << "] Layer " << layers_[k] << " Type " << types_[k] << " U:V "
                                  << moduleU_[k] << ":" << moduleV_[k];
}

void HGCalValidHexTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  std::vector<int> layers = {21, 21, 22, 22};
  std::vector<int> modU = {3, -3, 3, -3};
  std::vector<int> modV = {6, -6, 6, -6};
  std::vector<int> types = {2, 2, 2, 2};
  edm::ParameterSetDescription desc;
  desc.add<std::string>("NameDevice", "HGCal HE Silicon");
  desc.add<std::string>("NameSense", "HGCalHESiliconSensitive");
  desc.add<std::vector<int> >("Layers", layers);
  desc.add<std::vector<int> >("ModuleU", modU);
  desc.add<std::vector<int> >("ModuleV", modV);
  desc.add<std::vector<int> >("Types", types);
  descriptions.add("hgcalValidHexTesterHEF", desc);
}

// ------------ method called to produce the data  ------------
void HGCalValidHexTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const DetId::Detector det = (nameSense_ == "HGCalEESensitive") ? DetId::HGCalEE : DetId::HGCalHSi;
  const HGCalDDDConstants& hgdc = iSetup.getData(dddToken_);
  edm::LogVerbatim("HGCalGeom") << nameDetector_ << " Layers = " << hgdc.layers(true) << " Sectors = " << hgdc.sectors()
                                << "\n";
  for (unsigned int k = 0; k < layers_.size(); ++k) {
    int nCells = (types_[k] == 0) ? HGCSiliconDetId::HGCalFineN : HGCSiliconDetId::HGCalCoarseN;
    int ncell(0);
    for (int u = 0; u < 2 * nCells; ++u) {
      for (int v = 0; v < 2 * nCells; ++v) {
        if (((v - u) < nCells) && (u - v) <= nCells) {
          std::string state =
              hgdc.isValidHex8(layers_[k], moduleU_[k], moduleV_[k], u, v, false) ? "within" : "outside of";
          edm::LogVerbatim("HGCalGeom") << "Cell[" << k << "," << ncell << "] "
                                        << HGCSiliconDetId(
                                               det, 1, types_[k], layers_[k], moduleU_[k], moduleV_[k], u, v)
                                        << " is " << state << " fiducial volume";
          ++ncell;
        }
      }
    }
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalValidHexTester);
