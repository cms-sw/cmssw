// -*- C++ -*-
//
// Package:    HGCalWaferInFileOrientation
// Class:      HGCalWaferInFileOrientation
//
/**\class HGCalWaferInFileOrientation HGCalWaferInFileOrientation.cc
 test/HGCalWaferInFileOrientation.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2021/10/07
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
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HGCalWaferInFileOrientation : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit HGCalWaferInFileOrientation(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override {}
  void endRun(edm::Run const&, edm::EventSetup const&) override {}

private:
  const std::string nameDetector_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> geomToken_;
  const std::vector<int> layers_;
  const std::vector<int> waferU_, waferV_, types_;
};

HGCalWaferInFileOrientation::HGCalWaferInFileOrientation(const edm::ParameterSet& iC)
    : nameDetector_(iC.getParameter<std::string>("detectorName")),
      geomToken_(esConsumes<HGCalGeometry, IdealGeometryRecord, edm::Transition::BeginRun>(
          edm::ESInputTag{"", nameDetector_})),
      layers_(iC.getParameter<std::vector<int>>("layers")),
      waferU_(iC.getParameter<std::vector<int>>("waferUs")),
      waferV_(iC.getParameter<std::vector<int>>("waferVs")),
      types_(iC.getParameter<std::vector<int>>("types")) {
  edm::LogVerbatim("HGCalGeom") << "Test Orientation for " << layers_.size() << " ID's in " << nameDetector_;
}

void HGCalWaferInFileOrientation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<int> layer = {33, 33, 33, 33, 33, 33};
  std::vector<int> waferU = {12, 12, -3, -3, -9, -9};
  std::vector<int> waferV = {3, 9, 9, -12, 3, -12};
  std::vector<int> type = {2, 2, 2, 2, 2, 2};
  desc.add<std::string>("detectorName", "HGCalHESiliconSensitive");
  desc.add<std::vector<int>>("layers", layer);
  desc.add<std::vector<int>>("waferUs", waferU);
  desc.add<std::vector<int>>("waferVs", waferV);
  desc.add<std::vector<int>>("types", type);
  descriptions.add("hgcalWaferInFileOrientation", desc);
}

void HGCalWaferInFileOrientation::beginRun(edm::Run const&, edm::EventSetup const& iSetup) {
  const auto& geomR = iSetup.getData(geomToken_);
  const HGCalGeometry* geom = &geomR;
  const auto& hgdc = geom->topology().dddConstants();

  edm::LogVerbatim("HGCalGeom") << "Check Wafer orientations in file for " << nameDetector_ << "\n";
  if (hgdc.waferHexagon8()) {
    DetId::Detector det = (nameDetector_ == "HGCalHESiliconSensitive") ? DetId::HGCalHSi : DetId::HGCalEE;
    static std::vector<std::string> types = {"F", "b", "g", "gm", "a", "d", "dm", "c", "am", "bm", "X"};
    int allG(0), badP(0), badR(0), badG(0);
    int layerOff = hgdc.getLayerOffset();
    for (unsigned int k = 0; k < layers_.size(); ++k) {
      int layer = layers_[k] - layerOff;
      int indx = HGCalWaferIndex::waferIndex(layer, waferU_[k], waferV_[k]);
      int part1 = std::get<1>(hgdc.waferFileInfoFromIndex(indx));
      int rotn1 = std::get<2>(hgdc.waferFileInfoFromIndex(indx));
      HGCSiliconDetId id(det, 1, types_[k], layer, waferU_[k], waferV_[k], 0, 0);
      if (geom->topology().validModule(id, 3)) {
        ++allG;
        int part2 = hgdc.waferTypeRotation(id.layer(), id.waferU(), id.waferV(), false, false).first;
        int rotn2 = hgdc.waferTypeRotation(id.layer(), id.waferU(), id.waferV(), false, false).second;
        int part3 = hgdc.waferTypeRotation(id.layer(), id.waferU(), id.waferV(), true, false).first;
        int rotn3 = hgdc.waferTypeRotation(id.layer(), id.waferU(), id.waferV(), true, false).second;
        bool partOK = (part1 == part2);
        bool rotnOK = (rotn1 == rotn2);
        if (!partOK)
          ++badP;
        if (!rotnOK)
          ++badR;
        if ((!partOK) || (!rotnOK)) {
          ++badG;
          std::string partx1 = (part1 < static_cast<int>(types.size())) ? types[part1] : "X";
          std::string partx2 = (part2 < static_cast<int>(types.size())) ? types[part2] : "X";
          std::string partx3 = (part3 < static_cast<int>(types.size())) ? types[part3] : "X";
          const auto& xy = hgdc.waferPosition(layer, waferU_[k], waferV_[k], true, false);
          edm::LogVerbatim("HGCalGeom") << "ID[" << k << "]: " << id << " (" << partx1 << ":" << partx2 << ":" << partx3
                                        << ", " << rotn1 << ":" << rotn2 << ":" << rotn3 << ") at ("
                                        << std::setprecision(4) << xy.first << ", " << xy.second << ", "
                                        << hgdc.waferZ(layer, true) << ") failure flag " << partOK << ":" << rotnOK;
        }
      }
    }
    edm::LogVerbatim("HGCalGeom") << "\n\nFinds " << badG << " (" << badP << ":" << badR << ") mismatch among " << allG
                                  << ":" << layers_.size() << " wafers\n";
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalWaferInFileOrientation);
