// -*- C++ -*-
//
// Package:    HGCalGeometry
// Class:      HGCalBadIdCheck
//
/**\class HGCalBadIdCheck HGCalBadIdCheck.cc
 test/HGCalBadIdCheck.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2026/01/15
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
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomUtils.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HGCalBadIdCheck : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit HGCalBadIdCheck(const edm::ParameterSet &);
  ~HGCalBadIdCheck() override = default;
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
};

HGCalBadIdCheck::HGCalBadIdCheck(const edm::ParameterSet &iC)
    : nameDetector_(iC.getParameter<std::string>("nameDetector")),
      fileName_(iC.getParameter<std::string>("fileName")),
      tok_hgcal_{esConsumes<HGCalGeometry, IdealGeometryRecord, edm::Transition::BeginRun>(
          edm::ESInputTag{"", nameDetector_})},
      dets_((nameDetector_ == "HGCalEESensitive") ? DetId::HGCalEE : DetId::HGCalHSi) {
  edm::LogVerbatim("HGCGeom") << "Provide details of bad cells for " << nameDetector_ << " from the file " << fileName_;
}

void HGCalBadIdCheck::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("nameDetector", "HGCalEESensitive");
  desc.add<std::string>("fileName", "D122FE.txt");
  descriptions.add("hgcalBadIdCheck", desc);
}

// ------------ method called to produce the data  ------------
void HGCalBadIdCheck::beginRun(edm::Run const &iRun, edm::EventSetup const &iSetup) {
  //initiating hgc Geometry
  edm::LogVerbatim("HGCGeom") << "Tries to initialize HGCalGeometry and HGCalDDDConstants for " << nameDetector_;
  const edm::ESHandle<HGCalGeometry> &hgcGeom = iSetup.getHandle(tok_hgcal_);
  if (hgcGeom.isValid()) {
    const HGCalGeometry *geom = hgcGeom.product();
    edm::LogVerbatim("HGCGeom") << "Loaded HGCalDDConstants for " << nameDetector_;
    if (!fileName_.empty()) {
      edm::FileInPath filetmp("Geometry/HGCalGeometry/data/" + fileName_);
      std::string fileName = filetmp.fullPath();
      std::ifstream fInput(fileName.c_str());
      std::string detn = (dets_ == DetId::HGCalEE) ? "EE" : "HSi";
      if (!fInput.good()) {
        edm::LogVerbatim("HGCGeom") << "Cannot open file " << fileName;
      } else {
        char buffer[80];
        while (fInput.getline(buffer, 80)) {
          std::vector<std::string> items = HGCalGeomUtils::splitString(std::string(buffer));
          int32_t detId = std::atoi(items[0].c_str());
          if (DetId(detId).det() == dets_) {
            HGCSiliconDetId id(detId);
            int32_t layer = id.layer();
            int32_t waferU = id.waferU();
            int32_t waferV = id.waferV();
            HGCalParameters::waferInfo info = geom->topology().dddConstants().waferInfo(layer, waferU, waferV);
            std::ostringstream st1;
            int32_t cellU = id.cellU();
            int32_t cellV = id.cellV();
            st1 << detn << " z " << id.zside() << " Layer " << layer << " Wafer " << waferU << ":" << waferV << " Cell "
                << cellU << ":" << cellV << " " << HGCalTypes::waferTypeX(info.type) << ":" << info.part << ":"
                << HGCalTypes::waferTypeX(info.part) << ":" << info.orient << ":" << info.cassette;
            for (unsigned int k = 1; k < items.size(); ++k)
              st1 << ":" << items[k];
            edm::LogVerbatim("HGCGeom") << st1.str();
          }
        }
        fInput.close();
      }
    }
  } else {
    edm::LogWarning("HGCGeom") << "Cannot initiate HGCalGeometry for " << nameDetector_ << std::endl;
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalBadIdCheck);
