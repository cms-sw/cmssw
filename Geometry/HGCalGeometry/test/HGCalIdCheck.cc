// -*- C++ -*-
//
// Package:    HGCalGeometry
// Class:      HGCalIdCheck
//
/**\class HGCalIdCheck HGCalIdCheck.cc
 test/HGCalIdCheck.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2025/11/18
//
//

// system include files
#include <fstream>
#include <iostream>
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
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomUtils.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HGCalIdCheck : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit HGCalIdCheck(const edm::ParameterSet &);
  ~HGCalIdCheck() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void beginJob() override {}
  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &iEvent, edm::EventSetup const &) override {}
  void endRun(edm::Run const &, edm::EventSetup const &) override {}
  void endJob() override {}

private:
  const std::string nameDetector_;
  const std::string fileName_, outFileName_;
  const int mode_, cog_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> tok_hgcal_;
  const DetId::Detector dets_;
  std::vector<DetId> detIds_;
};

HGCalIdCheck::HGCalIdCheck(const edm::ParameterSet &iC)
    : nameDetector_(iC.getParameter<std::string>("nameDetector")),
      fileName_(iC.getParameter<std::string>("fileName")),
      outFileName_(iC.getParameter<std::string>("outFileName")),
      mode_(iC.getParameter<int>("mode")),
      cog_(iC.getParameter<int>("cog")),
      tok_hgcal_{esConsumes<HGCalGeometry, IdealGeometryRecord, edm::Transition::BeginRun>(
          edm::ESInputTag{"", nameDetector_})},
      dets_((nameDetector_ == "HGCalEESensitive") ? DetId::HGCalEE : DetId::HGCalHSi) {
  edm::LogVerbatim("HGCGeom") << "Test validity of cells for " << nameDetector_ << " with inputs from " << fileName_
                              << " and mode " << mode_ << " cog " << cog_;

  if (!fileName_.empty()) {
    edm::FileInPath filetmp("Geometry/HGCalGeometry/data/" + fileName_);
    std::string fileName = filetmp.fullPath();
    std::ifstream fInput(fileName.c_str());
    if (!fInput.good()) {
      edm::LogVerbatim("HGCGeom") << "Cannot open file " << fileName;
    } else {
      char buffer[80];
      while (fInput.getline(buffer, 80)) {
        std::vector<std::string> items = HGCalGeomUtils::splitString(std::string(buffer));
        DetId::Detector det = static_cast<DetId::Detector>(std::atoi(items[0].c_str()));
        if (det == dets_) {
          int32_t zp = std::atoi(items[1].c_str());
          int32_t type = std::atoi(items[2].c_str());
          int32_t layer = std::atoi(items[3].c_str());
          int32_t waferU = std::atoi(items[4].c_str());
          int32_t waferV = std::atoi(items[5].c_str());
          int32_t cellU = std::atoi(items[6].c_str());
          int32_t cellV = std::atoi(items[7].c_str());
          HGCSiliconDetId detId(det, zp, type, layer, waferU, waferV, cellU, cellV);
          detIds_.emplace_back(DetId(detId));
          edm::LogVerbatim("HGCGeom") << "[" << detIds_.size() << "] " << det << ":" << zp << ":" << type << ":"
                                      << layer << ":" << waferU << ":" << waferV << ":" << cellU << ":" << cellV
                                      << " ==> " << detId;
        }
      }
      fInput.close();
    }
    edm::LogVerbatim("HGCGeom") << "Reads " << detIds_.size() << " ID's from " << fileName_;
  } else {
    edm::LogVerbatim("HGCGeom") << "No input file is given == will test all valid ids for " << dets_;
  }
}

void HGCalIdCheck::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("nameDetector", "HGCalHESiliconSensitive");
  desc.add<std::string>("fileName", "D120E.txt");
  desc.add<std::string>("outFileName", "");
  desc.add<int>("mode", 1);
  desc.add<int>("cog", 0);
  descriptions.add("hgcalIdCheck", desc);
}

// ------------ method called to produce the data  ------------
void HGCalIdCheck::beginRun(edm::Run const &iRun, edm::EventSetup const &iSetup) {
  //initiating hgc Geometry
  edm::LogVerbatim("HGCGeom") << "Tries to initialize HGCalGeometry and HGCalDDDConstants for " << nameDetector_;
  const edm::ESHandle<HGCalGeometry> &hgcGeom = iSetup.getHandle(tok_hgcal_);
  if (hgcGeom.isValid()) {
    const HGCalGeometry *geom = hgcGeom.product();
    edm::LogVerbatim("HGCGeom") << "Loaded HGCalDDConstants for " << nameDetector_;
    std::ofstream fout;
    int bad(0);
    if (!outFileName_.empty()) {
      edm::LogVerbatim("HGCGeom") << "Opens " << outFileName_ << " to list IDs in question";
      fout.open(outFileName_.c_str(), std::ofstream::out);
    }
    if (fileName_.empty()) {
      detIds_ = geom->getValidDetIds(dets_);
      edm::LogVerbatim("HGCGeom") << "Gets " << detIds_.size() << " valid ID's for detector " << dets_;
    }
    int cog = cog_ / 10;
    cog = 10 * cog;
    for (unsigned int k = 0; k < detIds_.size(); ++k) {
      std::ostringstream st1;
      HGCSiliconDetId id(detIds_[k]);
      GlobalPoint xy = geom->getPosition(id, cog);
      bool valid = geom->topology().valid(id);
      DetId idx = geom->getClosestCell(xy, true);
      GlobalPoint cell = geom->getPosition(idx, cog_);
      std::string ok = (id.rawId() == idx.rawId()) ? "OK" : "ERROR";
      st1 << "Old: " << id << " Valid " << valid << " New: " << HGCSiliconDetId(idx) << " === " << ok << " at " << xy;
      HGCSiliconDetId idn(idx);
      std::string c1 = (id.layer() == idn.layer()) ? "" : "Layer MisMatch";
      std::string c2 = ((id.waferU() == idn.waferU()) && (id.waferV() == idn.waferV())) ? "" : "Wafer Mismatch";
      std::string c3 = ((id.cellU() == idn.cellU()) && (id.cellV() == idn.cellV())) ? "" : "Cell Mismatch";
      edm::LogVerbatim("HGCGeom") << "Hit[" << k << "] " << st1.str() << " Position (" << cell.x() << ", " << cell.y()
                                  << ", " << cell.z() << ") " << c1 << " " << c2 << " " << c3;
      if ((!outFileName_.empty()) && (id.rawId() != idx.rawId())) {
        ++bad;
        if (mode_ > 0) {
          HGCalParameters::waferInfo info =
              geom->topology().dddConstants().waferInfo(id.layer(), id.waferU(), id.waferV());
          fout << id.rawId() << " z " << id.zside() << " Layer " << id.layer() << " Wafer " << id.waferU() << ":"
               << id.waferV() << " Cell " << id.cellU() << ":" << id.cellV() << " " << HGCalTypes::waferTypeX(info.type)
               << ":" << info.part << ":" << HGCalTypes::waferTypeX(info.part) << ":" << info.orient << ":"
               << info.cassette << " " << c1 << " " << c2 << " " << c3 << std::endl;
        } else {
          fout << id.rawId() << " " << c1 << " " << c2 << " " << c3 << std::endl;
        }
      }
    }
    if (!outFileName_.empty()) {
      fout.close();
      edm::LogVerbatim("HGCGeom") << "Log information of " << bad << " problemaic IDs ";
    }
  } else {
    edm::LogWarning("HGCGeom") << "Cannot initiate HGCalGeometry for " << nameDetector_ << std::endl;
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalIdCheck);
