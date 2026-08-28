// -*- C++ -*-
//
// Package:    HGCalGeometry
// Class:      HGCalNeighbourTester
//
/**\class HGCalNeighbourTester HGCalNeighbourTester.cc
 test/HGCalNeighbourTester.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2026/01/27
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
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/transform.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "Geometry/CaloTopology/interface/HGCalNeighbourFinder.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomUtils.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HGCalNeighbourTester : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit HGCalNeighbourTester(const edm::ParameterSet &);
  ~HGCalNeighbourTester() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void beginJob() override {}
  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &iEvent, edm::EventSetup const &) override {}
  void endRun(edm::Run const &, edm::EventSetup const &) override {}
  void endJob() override {}

private:
  const std::string nameDetector_;
  const std::string fileName_;
  const int nskip_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> tok_hgcal_;
  const DetId::Detector dets_;
  std::vector<DetId> detIds_;
};

HGCalNeighbourTester::HGCalNeighbourTester(const edm::ParameterSet &iC)
    : nameDetector_(iC.getParameter<std::string>("nameDetector")),
      fileName_(iC.getParameter<std::string>("fileName")),
      nskip_(iC.getParameter<int>("nSkip")),
      tok_hgcal_{esConsumes<HGCalGeometry, IdealGeometryRecord, edm::Transition::BeginRun>(
          edm::ESInputTag{"", nameDetector_})},
      dets_((nameDetector_ == "HGCalEESensitive") ? DetId::HGCalEE : DetId::HGCalHSi) {
  edm::LogVerbatim("HGCalGeom") << "Test neighbours of cells for " << nameDetector_ << " with inputs from "
                                << fileName_;

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
        if (items.size() == 5) {
          int layer = std::atoi(items[0].c_str());
          int waferU = std::atoi(items[1].c_str());
          int waferV = std::atoi(items[2].c_str());
          int cellU = std::atoi(items[3].c_str());
          int cellV = std::atoi(items[4].c_str());
          DetId id1 = HGCSiliconDetId(dets_, 1, 0, layer, waferU, waferV, cellU, cellV);
          detIds_.emplace_back(id1);
          DetId id2 = HGCSiliconDetId(dets_, -1, 0, layer, waferU, waferV, cellU, cellV);
          detIds_.emplace_back(id2);
        }
      }
      fInput.close();
    }
  }

  if (detIds_.empty()) {
    edm::LogVerbatim("HGCalGeom") << "It will test all valid ids for " << dets_ << " skipping " << nskip_ << " entries";
  } else {
    edm::LogVerbatim("HGCalGeom") << "It will test for " << detIds_.size() << " cells from  " << fileName_;
  }
}

void HGCalNeighbourTester::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("nameDetector", "HGCalHESiliconSensitive");
  desc.add<std::string>("fileName", "D120NE.txt");
  desc.add<int>("nSkip", 1000);
  descriptions.add("hgcalNeighbourTester", desc);
}

// ------------ method called to produce the data  ------------
void HGCalNeighbourTester::beginRun(edm::Run const &iRun, edm::EventSetup const &iSetup) {
  //initiating hgc Geometry
  edm::LogVerbatim("HGCalGeom") << "Tries to initialize HGCalGeometry and HGCalDDDConstants for " << nameDetector_;
  const edm::ESHandle<HGCalGeometry> &hgcGeom = iSetup.getHandle(tok_hgcal_);
  if (hgcGeom.isValid()) {
    const HGCalGeometry *geom = hgcGeom.product();
    edm::LogVerbatim("HGCalGeom") << "Loaded HGCalDDConstants for " << nameDetector_;
    std::vector<DetId> detIds;
    uint32_t nskip(nskip_);
    if (detIds_.empty()) {
      detIds = geom->getValidDetIds(dets_);
    } else {
      static constexpr uint32_t mask = 0xFFFFFF;
      std::vector<DetId> allIds = geom->getValidDetIds(dets_);
      for (auto const &id : detIds_) {
        for (auto const &idz : allIds) {
          if ((id & mask) == (idz & mask)) {
            detIds.emplace_back(idz);
            break;
          }
        }
      }
      nskip = 1;
    }
    edm::LogVerbatim("HGCalGeom") << "Gets " << detIds.size() << " valid ID's for detector " << dets_;
    std::unique_ptr<HGCalNeighbourFinder> finder = std::make_unique<HGCalNeighbourFinder>(geom);
    for (unsigned int k = 0; k < detIds.size(); k += nskip) {
      edm::LogVerbatim("HGCalGeom") << "HGCalNeighbourTester for entry ** " << k << " *****";
      HGCSiliconDetId id(detIds[k]);
      std::vector<uint32_t> ids = finder->nearestNeighboursOfDetId(id.rawId());
      unsigned int nn(0);
      for (auto const &idZ : ids)
        if (idZ != 0) 
	  if (geom->validDetId(DetId(idZ)))
	    ++nn;
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] Layer " << id.layer() << " Wafer " << id.waferU() << ":"
                                    << id.waferV() << " Cell " << id.cellU() << ":" << id.cellV() << " has " << nn
                                    << " neighbours:";
      unsigned int k1(0);
      for (auto const &idZ : ids) {
        if (idZ != 0) {
          HGCSiliconDetId idx(idZ);
	  if (geom->validDetId(idx)) {
	    edm::LogVerbatim("HGCalGeom") << "[" << k1 << "] Layer " << idx.layer() << " Wafer " << idx.waferU() << ":"
					  << idx.waferV() << " Cell " << idx.cellU() << ":" << idx.cellV();
	    ++k1;
	  }
        }
      }
    }
  } else {
    edm::LogWarning("HGCalGeom") << "Cannot initiate HGCalGeometry for " << nameDetector_ << std::endl;
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalNeighbourTester);
