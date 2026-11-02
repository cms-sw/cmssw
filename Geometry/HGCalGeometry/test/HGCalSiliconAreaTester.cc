// -*- C++ -*-
//
// Package:    HGCalGeometry
// Class:      HGCalSiliconAreaTester
//
/**\class HGCalSiliconAreaTester HGCalSiliconAreaTester.cc
 test/HGCalSiliconAreaTester.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2026/10/29
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
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HGCalSiliconAreaTester : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit HGCalSiliconAreaTester(const edm::ParameterSet &);
  ~HGCalSiliconAreaTester() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void beginJob() override {}
  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &iEvent, edm::EventSetup const &) override {}
  void endRun(edm::Run const &, edm::EventSetup const &) override {}
  void endJob() override {}

private:
  const std::string nameDetector_;
  const int nskip_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> tok_hgcal_;
  const DetId::Detector dets_;
};

HGCalSiliconAreaTester::HGCalSiliconAreaTester(const edm::ParameterSet &iC)
    : nameDetector_(iC.getParameter<std::string>("nameDetector")),
      nskip_(iC.getParameter<int>("nSkip")),
      tok_hgcal_(esConsumes<HGCalGeometry, IdealGeometryRecord, edm::Transition::BeginRun>(
          edm::ESInputTag{"", nameDetector_})),
      dets_((nameDetector_ == "HGCalEESensitive") ? DetId::HGCalEE : DetId::HGCalHSi) {
  edm::LogVerbatim("HGCalGeom") << "Find area of cells for " << nameDetector_ << " of every " << nskip_
                                << " valid ones";
}

void HGCalSiliconAreaTester::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("nameDetector", "HGCalHESiliconSensitive");
  desc.add<int>("nSkip", 1000);
  descriptions.add("hgcalSiliconAreaTester", desc);
}

// ------------ method called to produce the data  ------------
void HGCalSiliconAreaTester::beginRun(edm::Run const &iRun, edm::EventSetup const &iSetup) {
  //initiating hgc Geometry
  edm::LogVerbatim("HGCalGeom") << "Tries to initialize HGCalGeometry and HGCalDDDConstants for " << nameDetector_;
  const edm::ESHandle<HGCalGeometry> &hgcGeom = iSetup.getHandle(tok_hgcal_);
  if (hgcGeom.isValid()) {
    const HGCalGeometry *geom = hgcGeom.product();
    std::vector<DetId> detIds = geom->getValidDetIds(dets_);
    edm::LogVerbatim("HGCalGeom") << "Loaded HGCalDDConstants for " << nameDetector_;
    for (unsigned int k = 0; k < detIds.size(); ++k) {
      std::ostringstream st1;
      HGCSiliconDetId id(detIds[k]);
      st1 << "[" << k << "] " << id << " Area from (Corners:Proper Subtractions) " << geom->getArea(detIds[k]) << ":"
          << geom->getAreaHex(detIds[k], true);
      edm::LogVerbatim("HGCalGeom") << st1.str();
    }
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalSiliconAreaTester);
