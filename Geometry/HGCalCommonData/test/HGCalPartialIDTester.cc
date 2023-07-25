// -*- C++ -*-
//
// Package:    HGCalPartialIDTester
// Class:      HGCalPartialIDTester
//
/**\class HGCalPartialIDTester HGCalPartialIDTester.cc
 test/HGCalPartialIDTester.cc

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
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomUtils.h"

class HGCalPartialIDTester : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit HGCalPartialIDTester(const edm::ParameterSet &);
  ~HGCalPartialIDTester() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void beginJob() override {}
  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &iEvent, edm::EventSetup const &) override {}
  void endRun(edm::Run const &, edm::EventSetup const &) override {}
  void endJob() override {}

private:
  const std::string nameDetector_;
  const std::string fileName_;
  const bool debug_;
  const edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> tok_hgcal_;
  const HGCalDDDConstants *hgcCons_;
  std::vector<int> zside_, layer_;
  std::vector<double> xpos_, ypos_;
};

HGCalPartialIDTester::HGCalPartialIDTester(const edm::ParameterSet &iC)
    : nameDetector_(iC.getParameter<std::string>("nameDetector")),
      fileName_(iC.getParameter<std::string>("fileName")),
      debug_(iC.getParameter<bool>("debug")),
      tok_hgcal_(esConsumes<HGCalDDDConstants, IdealGeometryRecord, edm::Transition::BeginRun>(edm::ESInputTag{"", nameDetector_})) {
  edm::LogVerbatim("HGCGeom") << "Test DetId from position for " << nameDetector_ << " with inputs from " << fileName_;
  if (!fileName_.empty()) {
    edm::FileInPath filetmp("Geometry/HGCalCommonData/data/" + fileName_);
    std::string fileName = filetmp.fullPath();
    std::ifstream fInput(fileName.c_str());
    if (!fInput.good()) {
      edm::LogVerbatim("HGCalGeom") << "Cannot open file " << fileName;
    } else {
      char buffer[80];
      const DetId::Detector dets = (nameDetector_ == "HGCalEESensitive") ? DetId::HGCalEE : DetId::HGCalHSi;
      while (fInput.getline(buffer, 80)) {
        std::vector<std::string> items = HGCalGeomUtils::splitString(std::string(buffer));
        if (items.size() == 5) {
          DetId::Detector det = static_cast<DetId::Detector>(std::atoi(items[0].c_str()));
	  if (det == dets) {
            if ((det == DetId::HGCalEE) || (det == DetId::HGCalHSi)) {
              zside_.emplace_back(std::atoi(items[1].c_str()));
              layer_.emplace_back(std::atoi(items[2].c_str()));
	      xpos_.emplace_back(std::atof(items[3].c_str()));
              ypos_.emplace_back(std::atof(items[4].c_str()));
	    }
          }
        }
      }
      fInput.close();
    }
  }
  edm::LogVerbatim("HGCalGeom") << "Reads " << layer_.size() << " positions from " << fileName_;
  for (unsigned int k = 0; k < layer_.size(); ++k) {
    edm::LogVerbatim("HGCalGeom") << "[" << k << "] zside " << zside_[k] << " Layer " << layer_[k] << " position " << xpos_[k] << ":" << ypos_[k];
  }
}

void HGCalPartialIDTester::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("nameDetector", "HGCalEESensitive");
  desc.add<std::string>("fileName", "partialD98.txt");
  desc.add<bool>("debug", true);
  descriptions.add("hgcalPartialIDTesterEE", desc);
}

// ------------ method called to produce the data  ------------
void HGCalPartialIDTester::beginRun(edm::Run const &iRun, edm::EventSetup const &iSetup) {
  //initiating hgc Geometry
  const edm::ESHandle<HGCalDDDConstants> &hgcCons = iSetup.getHandle(tok_hgcal_);
  if (!hgcCons.isValid()) {
      edm::LogWarning("HGCGeom") << "Cannot initiate HGCalDDDConstants for " << nameDetector_;
  } else {
    hgcCons_ = hgcCons.product();
    const DetId::Detector dets = (nameDetector_ == "HGCalEESensitive") ? DetId::HGCalEE : DetId::HGCalHSi;
    for (uint32_t i = 0; i < layer_.size(); i++) {
      int waferU, waferV, cellU, cellV, waferType;
      double wt;
      hgcCons_->waferFromPosition(xpos_[i], ypos_[i], zside_[i], layer_[i], waferU, waferV, cellU, cellV, waferType, wt, false, debug_);
      HGCalParameters::waferInfo info = hgcCons_->waferInfo(layer_[i], waferU, waferV);
      edm::LogVerbatim("HGCalGeom") << "Input " << dets << ":" << zside_[i] << ":"  << layer_[i] << ":" << xpos_[i] << ":" << ypos_[i] << " WaferType " << waferType << " Wafer " << waferU << ":" << waferV << " Cell " << cellU << ":" << cellV << " wt " << wt << " part:orien:cass " << info.part << ":" << info.orient << ":" << info.cassette;
      if (zside_[i] == -1) {
	hgcCons_->waferFromPosition(-xpos_[i], ypos_[i], zside_[i], layer_[i], waferU, waferV, cellU, cellV, waferType, wt, false, debug_);
	info = hgcCons_->waferInfo(layer_[i], waferU, waferV);
	edm::LogVerbatim("HGCalGeom") << "Input " << dets << ":" << zside_[i] << ":"  << layer_[i] << ":" << -xpos_[i] << ":" << ypos_[i] << " WaferType " << waferType << " Wafer " << waferU << ":" << waferV << " Cell " << cellU << ":" << cellV << " wt " << wt << " part:orien:cass " << info.part << ":" << info.orient << ":" << info.cassette;
      }
    }
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalPartialIDTester);
