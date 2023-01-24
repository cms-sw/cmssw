// -*- C++ -*-
//
// Package:    HGCalValidityTester
// Class:      HGCalValidityTester
//
/**\class HGCalValidityTester HGCalValidityTester.cc
 test/HGCalValidityTester.cc

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
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HGCalValidityTester : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit HGCalValidityTester(const edm::ParameterSet &);
  ~HGCalValidityTester() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void beginJob() override {}
  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &iEvent, edm::EventSetup const &) override {}
  void endRun(edm::Run const &, edm::EventSetup const &) override {}
  void endJob() override {}

private:
  const std::vector<std::string> nameDetectors_;
  const std::string fileName_;
  const std::vector<edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord>> tok_hgcal_;
  std::vector<const HGCalDDDConstants *> hgcCons_;
  std::vector<std::pair<DetId, uint32_t>> detIds_;
};

HGCalValidityTester::HGCalValidityTester(const edm::ParameterSet &iC)
    : nameDetectors_(iC.getParameter<std::vector<std::string>>("nameDetectors")),
      fileName_(iC.getParameter<std::string>("fileName")),
      tok_hgcal_{edm::vector_transform(nameDetectors_, [this](const std::string &name) {
        return esConsumes<HGCalDDDConstants, IdealGeometryRecord, edm::Transition::BeginRun>(edm::ESInputTag{"", name});
      })} {
  std::ostringstream st1;
  for (const auto &name : nameDetectors_)
    st1 << " : " << name;
  edm::LogVerbatim("HGCGeom") << "Test validity of cells for " << nameDetectors_.size() << " detectors" << st1.str()
                              << " with inputs from " << fileName_;
  if (!fileName_.empty()) {
    edm::FileInPath filetmp("Geometry/HGCalCommonData/data/" + fileName_);
    std::string fileName = filetmp.fullPath();
    std::ifstream fInput(fileName.c_str());
    if (!fInput.good()) {
      edm::LogVerbatim("HGCGeom") << "Cannot open file " << fileName;
    } else {
      char buffer[80];
      uint32_t lines(0);
      const std::vector<DetId::Detector> dets = {DetId::HGCalEE, DetId::HGCalHSi, DetId::HGCalHSc};
      while (fInput.getline(buffer, 80)) {
        ++lines;
        std::vector<std::string> items = HGCalGeomUtils::splitString(std::string(buffer));
        if (items.size() == 8) {
          DetId::Detector det = static_cast<DetId::Detector>(std::atoi(items[0].c_str()));
          auto itr = std::find(dets.begin(), dets.end(), det);
          if (itr != dets.end()) {
            uint32_t pos = static_cast<uint32_t>(itr - dets.begin());
            DetId id(0);
            if ((det == DetId::HGCalEE) || (det == DetId::HGCalHSi)) {
              int type = std::atoi(items[1].c_str());
              int zside = std::atoi(items[2].c_str());
              int layer = std::atoi(items[3].c_str());
              int waferU = std::atoi(items[4].c_str());
              int waferV = std::atoi(items[5].c_str());
              int cellU = std::atoi(items[6].c_str());
              int cellV = std::atoi(items[7].c_str());
              id = static_cast<DetId>(HGCSiliconDetId(det, zside, type, layer, waferU, waferV, cellU, cellV));
            } else if (det == DetId::HGCalHSc) {
              bool trig = (std::atoi(items[1].c_str()) > 0);
              int type = std::atoi(items[2].c_str());
              int zside = std::atoi(items[3].c_str());
              int sipm = std::atoi(items[4].c_str());
              int layer = std::atoi(items[5].c_str());
              int ring = std::atoi(items[6].c_str());
              int iphi = std::atoi(items[7].c_str());
              id = static_cast<DetId>(HGCScintillatorDetId(type, layer, zside * ring, iphi, trig, sipm));
            }
            if (id.rawId() != 0)
              detIds_.emplace_back(id, pos);
          }
        }
      }
      fInput.close();
    }
  }
  edm::LogVerbatim("HGCGeom") << "Reads " << detIds_.size() << " ID's from " << fileName_;
  for (unsigned int k = 0; k < detIds_.size(); ++k) {
    if ((detIds_[k].first).det() == DetId::HGCalHSc)
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << HGCScintillatorDetId(detIds_[k].first) << " from DDConstant "
                                    << (detIds_[k].second);
    else
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << HGCSiliconDetId(detIds_[k].first) << " from DDConstant "
                                    << (detIds_[k].second);
  }
}

void HGCalValidityTester::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  std::vector<std::string> names = {"HGCalEESensitive", "HGCalHESiliconSensitive", "HGCalHEScintillatorSensitive"};
  edm::ParameterSetDescription desc;
  desc.add<std::vector<std::string>>("nameDetectors", names);
  desc.add<std::string>("fileName", "missD88.txt");
  descriptions.add("hgcalValidityTester", desc);
}

// ------------ method called to produce the data  ------------
void HGCalValidityTester::beginRun(edm::Run const &iRun, edm::EventSetup const &iSetup) {
  //initiating hgc Geometry
  std::vector<std::string> names = {"HGCalEESensitive", "HGCalHESiliconSensitive", "HGCalHEScintillatorSensitive"};
  std::vector<DetId::Detector> dets = {DetId::HGCalEE, DetId::HGCalHSi, DetId::HGCalHSc};
  std::map<DetId::Detector, uint32_t> detMap;
  for (uint32_t i = 0; i < nameDetectors_.size(); i++) {
    edm::LogVerbatim("HGCGeom") << "Tries to initialize HGCalGeometry and HGCalDDDConstants for " << i << ":"
                                << nameDetectors_[i];
    const edm::ESHandle<HGCalDDDConstants> &hgcCons = iSetup.getHandle(tok_hgcal_[i]);
    if (hgcCons.isValid()) {
      hgcCons_.push_back(hgcCons.product());
    } else {
      edm::LogWarning("HGCGeom") << "Cannot initiate HGCalDDDConstants for " << nameDetectors_[i] << std::endl;
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
    std::pair<float, float> xy;
    bool valid(false);
    int layer(0), zside(1);
    const HGCalDDDConstants *cons = hgcCons_[detMap[(detIds_[k].first).det()]];
    if ((detIds_[k].first).det() == DetId::HGCalHSc) {
      HGCScintillatorDetId id(detIds_[k].first);
      layer = id.layer();
      zside = id.zside();
      xy = cons->locateCellTrap(zside, layer, id.ring(), id.iphi(), true, false);
      double z = zside * (cons->waferZ(layer, true));
      valid = cons->isValidTrap(zside, layer, id.ring(), id.iphi());
      auto cell = cons->assignCellTrap(zside * xy.first, xy.second, z, layer, true);
      HGCScintillatorDetId newId(cell[2], layer, zside * cell[0], cell[1], id.trigger(), id.sipm());
      st1 << "Old: " << id << " New: " << newId << " OK " << (id.rawId() == newId.rawId());
    } else {
      HGCSiliconDetId id(detIds_[k].first);
      layer = id.layer();
      zside = id.zside();
      xy = cons->locateCell(zside, layer, id.waferU(), id.waferV(), id.cellU(), id.cellV(), true, true, false, false);
      valid = cons->isValidHex8(layer, id.waferU(), id.waferV(), id.cellU(), id.cellV(), false);
      auto cell = cons->assignCellHex(zside * xy.first, xy.second, zside, layer, true, false, false);
      HGCSiliconDetId newId(id.det(), id.zside(), cell[2], id.layer(), cell[0], cell[1], cell[3], cell[4]);
      st1 << "Old: " << id << " New: " << newId << " OK " << (id.rawId() == newId.rawId());
    }
    double r = std::sqrt(xy.first * xy.first + xy.second * xy.second);
    double z = zside * (cons->waferZ(layer, true));
    auto range = cons->getRangeR(layer, true);
    edm::LogVerbatim("HGCalMiss") << "Hit[" << k << "] " << st1.str() << " Position (" << xy.first << ", " << xy.second
                                  << ", " << z << ") Valid " << valid << " R " << r << " (" << range.first << ":"
                                  << range.second << ")";
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalValidityTester);
