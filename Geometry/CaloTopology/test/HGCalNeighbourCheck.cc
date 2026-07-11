// -*- C++ -*-
//
// Package:    HGCalGeometry
// Class:      HGCalNeighbourCheck
//
/**\class HGCalNeighbourCheck HGCalNeighbourCheck.cc
 test/HGCalNeighbourCheck.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2026/03/05
//
//

// system include files
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/transform.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomUtils.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "Geometry/CaloTopology/interface/HGCalNeighbourFinder.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HGCalNeighbourCheck : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit HGCalNeighbourCheck(const edm::ParameterSet &);
  ~HGCalNeighbourCheck() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void beginJob() override {}
  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &iEvent, edm::EventSetup const &) override {}
  void endRun(edm::Run const &, edm::EventSetup const &) override {}
  void endJob() override {}

private:
  const std::string nameDetector_;
  const std::string fileName_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> tok_hgcal_;
  const DetId::Detector dets_;
  std::vector<DetId> detIds_;
};

HGCalNeighbourCheck::HGCalNeighbourCheck(const edm::ParameterSet &iC)
    : nameDetector_(iC.getParameter<std::string>("nameDetector")),
      fileName_(iC.getParameter<std::string>("fileName")),
      tok_hgcal_{esConsumes<HGCalGeometry, IdealGeometryRecord, edm::Transition::BeginRun>(
          edm::ESInputTag{"", nameDetector_})},
      dets_((nameDetector_ == "HGCalEESensitive") ? DetId::HGCalEE : DetId::HGCalHSi) {
  edm::LogVerbatim("HGCalGeom") << "Test validity of cells for " << nameDetector_ << " with inputs from " << fileName_;

  if (!fileName_.empty()) {
    edm::FileInPath filetmp("Geometry/CaloTopology/data/" + fileName_);
    std::string fileName = filetmp.fullPath();
    std::ifstream fInput(fileName.c_str());
    if (!fInput.good()) {
      edm::LogVerbatim("HGCalGeom") << "Cannot open file " << fileName;
    } else {
      char buffer[80];
      while (fInput.getline(buffer, 80)) {
        std::vector<std::string> items = HGCalGeomUtils::splitString(std::string(buffer));
        unsigned int id = static_cast<unsigned int>(std::atoi(items[0].c_str()));
        DetId::Detector det = DetId(id).det();
        if (det == dets_) {
          auto itr = std::find(detIds_.begin(), detIds_.end(), DetId(id));
          if (itr == detIds_.end()) {
            detIds_.emplace_back(DetId(id));
            edm::LogVerbatim("HGCalGeom") << "[" << detIds_.size() << "] " << HGCSiliconDetId(id);
          }
        }
      }
      fInput.close();
    }
    edm::LogVerbatim("HGCalGeom") << "Reads " << detIds_.size() << " ID's from " << fileName_;
  } else {
    edm::LogVerbatim("HGCalGeom") << "No input file is given == will test all valid ids for " << dets_;
  }
}

void HGCalNeighbourCheck::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("nameDetector", "HGCalHESiliconSensitive");
  desc.add<std::string>("fileName", "D120E.txt");
  descriptions.add("hgcalNeighbourCheck", desc);
}

// ------------ method called to produce the data  ------------
void HGCalNeighbourCheck::beginRun(edm::Run const &iRun, edm::EventSetup const &iSetup) {
  //initiating hgc Geometry
  edm::LogVerbatim("HGCalGeom") << "Tries to initialize HGCalGeometry and HGCalDDDConstants for " << nameDetector_;
  const edm::ESHandle<HGCalGeometry> &hgcGeom = iSetup.getHandle(tok_hgcal_);
  if (hgcGeom.isValid()) {
    const HGCalGeometry *geom = hgcGeom.product();
    edm::LogVerbatim("HGCalGeom") << "Loaded HGCalDDConstants for " << nameDetector_;
    if (fileName_.empty()) {
      detIds_ = geom->getValidDetIds(dets_);
      edm::LogVerbatim("HGCalGeom") << "Gets " << detIds_.size() << " valid ID's for detector " << dets_;
    }
    std::unique_ptr<HGCalNeighbourFinder> finder = std::make_unique<HGCalNeighbourFinder>(geom->topology().dddConstants());
    for (unsigned int k = 0; k < detIds_.size(); ++k) {
      std::ostringstream st1;
      HGCSiliconDetId id(detIds_[k]);
      std::vector<uint32_t> ids = finder->nearestNeighboursOfDetId(id.rawId());
      unsigned int nn(0);
      for (auto const &idZ : ids)
        if (idZ != 0)
          ++nn;
      st1 << "[" << k << "]" << id << " with " << nn << " neighbours:";
      for (auto &idx : ids) {
	if (idx != 0) 
	  st1 << "(" << HGCSiliconDetId(idx).waferU() << "," << HGCSiliconDetId(idx).waferV() << ","
	      << HGCSiliconDetId(idx).cellU() << "," << HGCSiliconDetId(idx).cellV() << ")";
      }
      edm::LogVerbatim("HGCalGeom") << st1.str();
    }
    edm::LogVerbatim("HGCalGeom") << "Log information of " << detIds_.size() << " cells";
  } else {
    edm::LogWarning("HGCalGeom") << "Cannot initiate HGCalGeometry for " << nameDetector_ << std::endl;
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalNeighbourCheck);
