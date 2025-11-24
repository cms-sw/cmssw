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
  const std::vector<std::string> nameDetectors_;
  const std::string fileName_;
  const std::vector<edm::ESGetToken<HGCalGeometry, IdealGeometryRecord>> tok_hgcal_;
  std::vector<const HGCalGeometry *> hgcGeom_;
  std::vector<std::pair<HGCSiliconDetId, uint32_t>> detIds_;
};

HGCalIdCheck::HGCalIdCheck(const edm::ParameterSet &iC)
    : nameDetectors_(iC.getParameter<std::vector<std::string>>("nameDetectors")),
      fileName_(iC.getParameter<std::string>("fileName")),
      tok_hgcal_{edm::vector_transform(nameDetectors_, [this](const std::string &name) {
        return esConsumes<HGCalGeometry, IdealGeometryRecord, edm::Transition::BeginRun>(edm::ESInputTag{"", name});
      })} {
  std::ostringstream st1;
  for (const auto &name : nameDetectors_)
    st1 << " : " << name;
  edm::LogVerbatim("HGCGeom") << "Test validity of cells for " << nameDetectors_.size() << " detectors" << st1.str()
                              << " with inputs from " << fileName_;

  const std::vector<DetId::Detector> dets = {DetId::HGCalEE, DetId::HGCalHSi};
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
        int32_t zp = std::atoi(items[2].c_str());
        int32_t type = std::atoi(items[1].c_str());
        int32_t layer = std::atoi(items[3].c_str());
        int32_t waferU = std::atoi(items[4].c_str());
        int32_t waferV = std::atoi(items[5].c_str());
        int32_t cellU = std::atoi(items[6].c_str());
        int32_t cellV = std::atoi(items[7].c_str());
        HGCSiliconDetId detId(det, zp, type, layer, waferU, waferV, cellU, cellV);
        auto itr = std::find(dets.begin(), dets.end(), det);
        if (itr != dets.end()) {
          uint32_t pos = static_cast<uint32_t>(itr - dets.begin());
          detIds_.emplace_back(std::pair<HGCSiliconDetId, int>(detId, pos));
        }
      }
      fInput.close();
    }
  }
  edm::LogVerbatim("HGCGeom") << "Reads " << detIds_.size() << " ID's from " << fileName_;
}

void HGCalIdCheck::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  std::vector<std::string> names = {"HGCalEESensitive", "HGCalHESiliconSensitive"};
  edm::ParameterSetDescription desc;
  desc.add<std::vector<std::string>>("nameDetectors", names);
  desc.add<std::string>("fileName", "D120E.txt");
  descriptions.add("hgcalIdCheck", desc);
}

// ------------ method called to produce the data  ------------
void HGCalIdCheck::beginRun(edm::Run const &iRun, edm::EventSetup const &iSetup) {
  //initiating hgc Geometry
  std::vector<std::string> names = {"HGCalEESensitive", "HGCalHESiliconSensitive"};
  std::vector<DetId::Detector> dets = {DetId::HGCalEE, DetId::HGCalHSi};
  std::map<DetId::Detector, uint32_t> detMap;
  for (uint32_t i = 0; i < nameDetectors_.size(); i++) {
    edm::LogVerbatim("HGCGeom") << "Tries to initialize HGCalGeometry and HGCalDDDConstants for " << i << ":"
                                << nameDetectors_[i];
    const edm::ESHandle<HGCalGeometry> &hgcGeom = iSetup.getHandle(tok_hgcal_[i]);
    if (hgcGeom.isValid()) {
      hgcGeom_.push_back(hgcGeom.product());
    } else {
      edm::LogWarning("HGCGeom") << "Cannot initiate HGCalGeometry for " << nameDetectors_[i] << std::endl;
    }
    auto ii = std::find(names.begin(), names.end(), nameDetectors_[i]);
    if (ii != names.end()) {
      uint32_t k = static_cast<uint32_t>(ii - names.begin());
      detMap[dets[k]] = i;
    }
  }
  edm::LogVerbatim("HGCGeom") << "Loaded HGCalDDConstants for " << detMap.size() << " detectors";

  for (auto itr = detMap.begin(); itr != detMap.end(); ++itr)
    edm::LogVerbatim("HGCGeom") << "[" << itr->second << "]: " << nameDetectors_[itr->second] << " for Detector "
                                << itr->first;

  for (unsigned int k = 0; k < detIds_.size(); ++k) {
    std::ostringstream st1;
    const HGCalGeometry *geom = hgcGeom_[detMap[(detIds_[k].first).det()]];
    HGCSiliconDetId id(detIds_[k].first);
    GlobalPoint xy = geom->getPosition(id, true);
    bool valid = geom->topology().valid(id);
    DetId idx = geom->getClosestCell(xy);
    GlobalPoint cell = geom->getPosition(idx, true);
    std::string ok = (id.rawId() == idx.rawId()) ? "OK" : "ERROR";
    st1 << "Old: " << id << " Valid " << valid << " New: " << HGCSiliconDetId(idx) << " === " << ok << " at " << xy;
    HGCSiliconDetId idn(idx);
    std::string c1 = (id.layer() == idn.layer()) ? "" : "Layer MisMatch";
    std::string c2 = ((id.waferU() == idn.waferU()) && (id.waferV() == idn.waferV())) ? "" : "Wafer Mismatch";
    std::string c3 = ((id.cellU() == idn.cellU()) && (id.cellV() == idn.cellV())) ? "" : "Cell Mismatch";
    edm::LogVerbatim("HGCGeom") << "Hit[" << k << "] " << st1.str() << " Position (" << cell.x() << ", " << cell.y()
                                << ", " << cell.z() << ") " << c1 << " " << c2 << " " << c3;
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalIdCheck);
