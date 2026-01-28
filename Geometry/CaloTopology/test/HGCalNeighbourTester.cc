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
  const int nskip_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> tok_hgcal_;
  const DetId::Detector dets_;
  std::vector<DetId> detIds_;
};

HGCalNeighbourTester::HGCalNeighbourTester(const edm::ParameterSet &iC)
    : nameDetector_(iC.getParameter<std::string>("nameDetector")),
      nskip_(iC.getParameter<int>("nSkip")),
      tok_hgcal_{esConsumes<HGCalGeometry, IdealGeometryRecord, edm::Transition::BeginRun>(
          edm::ESInputTag{"", nameDetector_})},
      dets_((nameDetector_ == "HGCalEESensitive") ? DetId::HGCalEE : DetId::HGCalHSi) {
  edm::LogVerbatim("HGCGeom") << "Test neighbours of cells for " << nameDetector_;

  edm::LogVerbatim("HGCGeom") << "It will test all valid ids for " << dets_ << " skipping " << nskip_ << " entries";
}

void HGCalNeighbourTester::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("nameDetector", "HGCalHESiliconSensitive");
  desc.add<int>("nSkip", 1000);
  descriptions.add("hgcalNeighbourTester", desc);
}

// ------------ method called to produce the data  ------------
void HGCalNeighbourTester::beginRun(edm::Run const &iRun, edm::EventSetup const &iSetup) {
  //initiating hgc Geometry
  edm::LogVerbatim("HGCGeom") << "Tries to initialize HGCalGeometry and HGCalDDDConstants for " << nameDetector_;
  const edm::ESHandle<HGCalGeometry> &hgcGeom = iSetup.getHandle(tok_hgcal_);
  if (hgcGeom.isValid()) {
    const HGCalGeometry *geom = hgcGeom.product();
    edm::LogVerbatim("HGCGeom") << "Loaded HGCalDDConstants for " << nameDetector_;
    detIds_ = geom->getValidDetIds(dets_);
    edm::LogVerbatim("HGCGeom") << "Gets " << detIds_.size() << " valid ID's for detector " << dets_;
    auto hgc_ = std::make_unique<HGCalNeighbourFinder>(&(geom->topology().dddConstants()));
    for (unsigned int k = 0; k < detIds_.size(); k += nskip_) {
      HGCSiliconDetId id(detIds_[k]);
      std::vector<unsigned int> ids = hgc_->nearestNeighboursOfDetId(detIds_[k].rawId());
      edm::LogVerbatim("HGCGeom") << "[" << k << "] Layer " << id.layer() << " Wafer " << id.waferU() << ":" << id.waferV() << " Cell " << id.cellU() << ":" << id.cellV() << " has " << ids.size() << " neighbours:";
      unsigned int k1(0);
      for (auto const& idZ : ids) {
	HGCSiliconDetId idx(idZ);
	edm::LogVerbatim("HGCGeom") << "[" << k << "] Layer " << id.layer() << " Wafer " << id.waferU() << ":" << id.waferV() << " Cell " << id.cellU() << ":" << id.cellV() << " has " << ids.size() << " neighbours:";
	++k1;
      }
    }
  } else {
    edm::LogWarning("HGCGeom") << "Cannot initiate HGCalGeometry for " << nameDetector_ << std::endl;
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalNeighbourTester);
