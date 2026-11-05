// -*- C++ -*-
//
// Package:    HGCalGeometry
// Class:      HGCalAreaTester
//
/**\class HGCalAreaTester HGCalAreaTester.cc
 test/HGCalAreaTester.cc

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
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomUtils.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "Geometry/HGCalGeometry/interface/HGCalNeighbourFinder.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HGCalAreaTester : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit HGCalAreaTester(const edm::ParameterSet &);
  ~HGCalAreaTester() override = default;
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

HGCalAreaTester::HGCalAreaTester(const edm::ParameterSet &iC)
    : nameDetector_(iC.getParameter<std::string>("nameDetector")),
      fileName_(iC.getParameter<std::string>("fileName")),
      tok_hgcal_(esConsumes<HGCalGeometry, IdealGeometryRecord, edm::Transition::BeginRun>(
          edm::ESInputTag{"", nameDetector_})),
      dets_((nameDetector_ == "HGCalEESensitive")
                ? DetId::HGCalEE
                : ((nameDetector_ == "HGCalHESilicnSensitive") ? DetId::HGCalHSi : DetId::HGCalHSc)) {
  edm::LogVerbatim("HGCalGeom") << "Test validity of cells for " << nameDetector_ << " with inputs from " << fileName_;

  if (!fileName_.empty()) {
    edm::FileInPath filetmp("Geometry/HGCalGeometry/data/" + fileName_);
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
        edm::LogVerbatim("HGCGeom") << "ID " << std::hex << id << std::dec << " Detector " << det;
        if (det == dets_) {
          auto itr = std::find(detIds_.begin(), detIds_.end(), DetId(id));
          if (itr == detIds_.end()) {
            detIds_.emplace_back(DetId(id));
            if (dets_ == DetId::HGCalHSc)
              edm::LogVerbatim("HGCGeom") << "[" << detIds_.size() << "] " << HGCScintillatorDetId(id);
            else
              edm::LogVerbatim("HGCGeom") << "[" << detIds_.size() << "] " << HGCSiliconDetId(id);
          }
        }
      }
      fInput.close();
    }
    edm::LogVerbatim("HGCalGeom") << "Reads " << detIds_.size() << " ID's from " << fileName_;
  }
  if (detIds_.empty()) {
    edm::LogVerbatim("HGCalGeom") << "List of DetIds not provided == will test all valid ids for " << dets_;
  }
}

void HGCalAreaTester::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("nameDetector", "HGCalHEScintillatorSensitive");
  desc.add<std::string>("fileName", "detID.txt");
  descriptions.add("hgcalAreaTester", desc);
}

// ------------ method called to produce the data  ------------
void HGCalAreaTester::beginRun(edm::Run const &iRun, edm::EventSetup const &iSetup) {
  //initiating hgc Geometry
  edm::LogVerbatim("HGCalGeom") << "Tries to initialize HGCalGeometry and HGCalDDDConstants for " << nameDetector_;
  const edm::ESHandle<HGCalGeometry> &hgcGeom = iSetup.getHandle(tok_hgcal_);
  if (hgcGeom.isValid()) {
    const HGCalGeometry *geom = hgcGeom.product();
    edm::LogVerbatim("HGCalGeom") << "Loaded HGCalDDConstants for " << nameDetector_;
    if (detIds_.empty()) {
      detIds_ = geom->getValidDetIds(dets_);
      edm::LogVerbatim("HGCalGeom") << "Gets " << detIds_.size() << " valid ID's for detector " << dets_;
    }
    std::unique_ptr<HGCalNeighbourFinder> finder = std::make_unique<HGCalNeighbourFinder>(geom);
    for (unsigned int k = 0; k < detIds_.size(); ++k) {
      if (geom->validDetId(detIds_[k])) {
        std::ostringstream st1;
        st1 << "[" << k << "] ";
        if (dets_ == DetId::HGCalHSc)
          st1 << HGCScintillatorDetId(detIds_[k]);
        else
          st1 << HGCSiliconDetId(detIds_[k]);
        std::vector<GlobalPoint> corner1 = geom->getCorners(detIds_[k]);
        std::vector<GlobalPoint> corner2 = geom->getNewCorners(detIds_[k], false);
        st1 << " Method 1:Area " << geom->getAreaX(detIds_[k]) << " with " << corner1.size() << " corners:";
        for (auto const &cor : corner1)
          st1 << " (" << cor.perp() << ":" << cor.x() << ":" << cor.y() << ":" << cor.z() << ")";
        st1 << " and Method 2: Area " << geom->getArea(detIds_[k]) << " with " << corner2.size() << " corners:";
        for (auto const &cor : corner2)
          st1 << " (" << cor.perp() << ":" << cor.x() << ":" << cor.y() << ":" << cor.z() << ")";
        edm::LogVerbatim("HGCalGeom") << st1.str();
      }
    }
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalAreaTester);
