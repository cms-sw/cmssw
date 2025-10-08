// -*- C++ -*-
//
// Package:    HGCalValidityTester
// Class:      HGCalMissingID
//
/**\class HGCalMissingID HGCalMissingID.cc
 test/HGCalMissingID.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2022/10/22
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

class HGCalMissingID : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit HGCalMissingID(const edm::ParameterSet &);
  ~HGCalMissingID() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void beginJob() override {}
  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &iEvent, edm::EventSetup const &) override {}
  void endRun(edm::Run const &, edm::EventSetup const &) override {}
  void endJob() override {}

private:
  const std::vector<std::string> nameDetectors_;
  const std::string fileName_;
  const int32_t nFirst_, nTot_;
  const std::vector<edm::ESGetToken<HGCalGeometry, IdealGeometryRecord>> tok_hgcal_;
  std::vector<const HGCalGeometry *> hgcGeom_;
  std::vector<std::pair<DetId, uint32_t>> detIds_;
};

HGCalMissingID::HGCalMissingID(const edm::ParameterSet &iC)
    : nameDetectors_(iC.getParameter<std::vector<std::string>>("nameDetectors")),
      fileName_(iC.getParameter<std::string>("fileName")),
      nFirst_(iC.getParameter<int>("first")),
      nTot_(iC.getParameter<int>("total")),
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
      int kount(0);
      while (fInput.getline(buffer, 80)) {
        std::vector<std::string> items = HGCalGeomUtils::splitString(std::string(buffer));
        unsigned int id = static_cast<unsigned int>(std::atoi(items[0].c_str()));
        DetId::Detector det = DetId(id).det();
        auto itr = std::find(dets.begin(), dets.end(), det);
        if (itr != dets.end()) {
          uint32_t pos = static_cast<uint32_t>(itr - dets.begin());
          ++kount;
          if (kount > nFirst_) {
            detIds_.emplace_back(std::pair<DetId, int>(id, pos));
            if ((detIds_.size() > static_cast<size_t>(nTot_)) && (nTot_ > 0))
              break;
          }
        }
      }
      fInput.close();
    }
  }
  edm::LogVerbatim("HGCGeom") << "Reads " << detIds_.size() << " ID's from " << fileName_;
  for (unsigned int i = 0; i < dets.size(); ++i) {
    int32_t nt(0);
    for (unsigned int k = 0; k < detIds_.size(); ++k) {
      if (detIds_[k].second == i)
        ++nt;
    }
    edm::LogVerbatim("HGCGeom") << "[" << i << "] " << nt << " IDs for " << dets[i];
  }
}

void HGCalMissingID::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  std::vector<std::string> names = {"HGCalEESensitive", "HGCalHESiliconSensitive"};
  edm::ParameterSetDescription desc;
  desc.add<std::vector<std::string>>("nameDetectors", names);
  desc.add<std::string>("fileName", "missingIDsV19.txt");
  desc.add<int>("first", 0);
  desc.add<int>("total", 10);
  descriptions.add("hgcalMissingID", desc);
}

// ------------ method called to produce the data  ------------
void HGCalMissingID::beginRun(edm::Run const &iRun, edm::EventSetup const &iSetup) {
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
    GlobalPoint xy = geom->getPosition(id, false, false);
    bool valid = geom->topology().valid(id);
    auto cell = geom->getPosition(DetId(id), false, false);
    DetId idx = geom->getClosestCell(cell);
    std::string ok = (id.rawId() == idx.rawId()) ? "OK" : "ERROR";
    st1 << "Old: " << id << " Valid " << valid << " New: " << HGCSiliconDetId(idx) << " === " << ok << " at " << xy;
    edm::LogVerbatim("HGCGeom") << "Hit[" << k << "] " << st1.str() << " Position (" << cell.x() << ", " << cell.y()
                                << ", " << cell.z() << ")";
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalMissingID);
