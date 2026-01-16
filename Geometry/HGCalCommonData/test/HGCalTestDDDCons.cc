// -*- C++ -*-
//
// Package:    HGCalTestDDDCons.cc
// Class:      HGCalTestDDDCons
//
/**\class HGCalTestDDDCons HGCalTestDDDCons.cc
 test/HGCalTestDDDCons.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Pruthvi Suryadevara
//         Created:  Mon 2025/7/11
//
//

// system include files
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

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
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/HGCalCell.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"

class HGCalTestDDDCons : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit HGCalTestDDDCons(const edm::ParameterSet &);
  ~HGCalTestDDDCons() override = default;
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
  int size;
  std::vector<const HGCalDDDConstants *> hgcCons_;
  std::vector<std::pair<DetId, uint32_t>> detIds_;
  std::vector<double> xwafer_, ywafer_, xcell_, ycell_, xcellOff_, ycellOff_;
};

HGCalTestDDDCons::HGCalTestDDDCons(const edm::ParameterSet &iC)
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
      char buffer[200];
      const std::vector<DetId::Detector> dets = {DetId::HGCalEE, DetId::HGCalHSi, DetId::HGCalHSc};
      while (fInput.getline(buffer, 200)) {
        std::vector<std::string> items = HGCalGeomUtils::splitString(std::string(buffer));
        if (items.size() == 14) {
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
              detIds_.emplace_back(id, pos);
              xwafer_.emplace_back(std::atof(items[8].c_str()));
              ywafer_.emplace_back(std::atof(items[9].c_str()));
              xcellOff_.emplace_back(std::atof(items[10].c_str()));
              ycellOff_.emplace_back(std::atof(items[11].c_str()));
              xcell_.emplace_back(std::atof(items[12].c_str()));
              ycell_.emplace_back(std::atof(items[13].c_str()));
            }
          }
        }
      }
      fInput.close();
    }
  }
  size = detIds_.size();
  edm::LogVerbatim("HGCGeom") << "Reads " << detIds_.size() << " ID's from " << fileName_;
}

void HGCalTestDDDCons::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  std::vector<std::string> names = {"HGCalEESensitive", "HGCalHESiliconSensitive"};
  edm::ParameterSetDescription desc;
  desc.add<std::vector<std::string>>("nameDetectors", names);
  desc.add<std::string>("fileName", "missD120.txt");
  descriptions.add("hgcalTestDDDCons", desc);
}

// ------------ method called to produce the data  ------------
void HGCalTestDDDCons::beginRun(edm::Run const &iRun, edm::EventSetup const &iSetup) {
  //initiating hgc Geometry
  std::vector<std::string> names = {"HGCalEESensitive", "HGCalHESiliconSensitive"};
  std::vector<DetId::Detector> dets = {DetId::HGCalEE, DetId::HGCalHSi};
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

  int cellU(0), cellV(0), waferType(-1), waferU(0), waferV(0);
  double wt(1.0);
  for (int k = 0; k < size; ++k) {
    const HGCalDDDConstants *cons = hgcCons_[detMap[(detIds_[k].first).det()]];
    HGCSiliconDetId id(detIds_[k].first);
    auto hgpar_ = cons->getParameter();
    HGCalCell celli(hgpar_->waferSize_, hgpar_->nCellsFine_, hgpar_->nCellsCoarse_);
    auto placement = cons->placementIndex(id);
    int ncell_ = id.lowDensity() ? hgpar_->nCellsCoarse_ : hgpar_->nCellsFine_;
    auto partialType = cons->partialWaferType(id.layer(), id.waferU(), id.waferV());
    auto cellType = HGCalCell::cellType(id.cellU(), id.cellV(), ncell_, placement, partialType);
    auto waferxy = cons->waferPositionWithCshift(id.layer(), id.waferU(), id.waferV(), true, true, false);
    auto cellxy_cog = cons->locateCell(id, true, false);
    auto cellxy_ncog = cons->locateCell(id, false, false);
    waferU = id.waferU();
    waferV = id.waferV();
    double xx = id.zside() * xcell_[k];
    cons->waferFromPosition(
        xx, ycell_[k], id.zside(), id.layer(), waferU, waferV, cellU, cellV, waferType, wt, false, false);
    auto valid = cons->isValidHex8(id.layer(), id.waferU(), id.waferV(), id.cellU(), id.cellV(), true);
    float scale = 0.1;
    edm::LogVerbatim("HGCGeom") << "Hit[" << k << "] " << id << " Valid " << valid
                                << " zside:layer:waferU:waferV:cellU:cellV " << id.layer() << ":" << id.waferU() << ":"
                                << id.waferV() << ":" << id.cellU() << ":" << id.cellV()
                                << " Observed coordinates wafer:cellCOG:cell " << waferxy.first << "," << waferxy.second
                                << ":" << cellxy_ncog.first << "," << cellxy_ncog.second << ":" << cellxy_cog.first
                                << "," << cellxy_cog.second << " CellType:CellPosition " << cellType.second << ":"
                                << cellType.first;
    if (std::sqrt(std::pow(waferxy.first + scale * xwafer_[k], 2) + std::pow(waferxy.second - scale * ywafer_[k], 2)) >
        0.01) {
      edm::LogVerbatim("HGCGeom") << " Error wafer mismatch actual:observed (" << xwafer_[k] << "," << ywafer_[k]
                                  << "):(" << waferxy.first << "," << waferxy.second << ") ";
    }
    if (std::sqrt(std::pow(cellxy_ncog.first + scale * xcell_[k], 2) +
                  std::pow(cellxy_ncog.second - scale * ycell_[k], 2)) > 0.01) {
      edm::LogVerbatim("HGCGeom") << " Error cell COG mismatch actual:observed (" << xcell_[k] << "," << ycell_[k]
                                  << "):(" << cellxy_ncog.first << "," << cellxy_ncog.second << ") ";
    }
    if (std::sqrt(std::pow(cellxy_cog.first + scale * xcellOff_[k], 2) +
                  std::pow(cellxy_cog.second - scale * ycellOff_[k], 2)) > 0.01) {
      edm::LogVerbatim("HGCGeom") << " Error cell center mismatch actual:observed (" << xcellOff_[k] << ","
                                  << ycellOff_[k] << "):(" << cellxy_cog.first << "," << cellxy_cog.second << ") ";
    }
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalTestDDDCons);
