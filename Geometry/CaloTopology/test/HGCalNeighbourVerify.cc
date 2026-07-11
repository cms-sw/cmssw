// -*- C++ -*-
//
// Package:    HGCalGeometry
// Class:      HGCalNeighbourVerify
//
/**\class HGCalNeighbourVerify HGCalNeighbourVerify.cc
 test/HGCalNeighbourVerify.cc

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

class HGCalNeighbourVerify : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit HGCalNeighbourVerify(const edm::ParameterSet &);
  ~HGCalNeighbourVerify() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  void beginJob() override {}
  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &iEvent, edm::EventSetup const &) override {}
  void endRun(edm::Run const &, edm::EventSetup const &) override {}
  void endJob() override {}

private:
  const std::string nameDetector_;
  uint32_t idUV_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> tok_hgcal_;
  const DetId::Detector dets_;
};

HGCalNeighbourVerify::HGCalNeighbourVerify(const edm::ParameterSet &iC)
    : nameDetector_(iC.getParameter<std::string>("nameDetector")),
      idUV_(0),
      tok_hgcal_{esConsumes<HGCalGeometry, IdealGeometryRecord, edm::Transition::BeginRun>(
          edm::ESInputTag{"", nameDetector_})},
      dets_((nameDetector_ == "HGCalEESensitive") ? DetId::HGCalEE : DetId::HGCalHSi) {
  int32_t waferU = iC.getParameter<int>("waferU");
  int32_t waferV = iC.getParameter<int>("waferV");
  int32_t cellU = iC.getParameter<int>("cellU");
  int32_t cellV = iC.getParameter<int>("cellV");
  int32_t waferUabs(std::abs(waferU)), waferVabs(std::abs(waferV));
  int32_t waferUsign = (waferU >= 0) ? 0 : 1;
  int32_t waferVsign = (waferV >= 0) ? 0 : 1;
  idUV_ |= (((cellU & HGCSiliconDetId::kHGCalCellUMask) << HGCSiliconDetId::kHGCalCellUOffset) |
	    ((cellV & HGCSiliconDetId::kHGCalCellVMask) << HGCSiliconDetId::kHGCalCellVOffset) |
            ((waferUabs & HGCSiliconDetId::kHGCalWaferUMask) << HGCSiliconDetId::kHGCalWaferUOffset) |
            ((waferUsign & HGCSiliconDetId::kHGCalWaferUSignMask) << HGCSiliconDetId::kHGCalWaferUSignOffset) |
            ((waferVabs & HGCSiliconDetId::kHGCalWaferVMask) << HGCSiliconDetId::kHGCalWaferVOffset) |
            ((waferVsign & HGCSiliconDetId::kHGCalWaferVSignMask) << HGCSiliconDetId::kHGCalWaferVSignOffset));

  edm::LogVerbatim("HGCalGeom") << "Test neighbours of cell (" << cellU << ", " << cellV << ") in wafer (" << waferU << ", " << waferV << " for " << nameDetector_ << " given by " << std::hex << idUV_ << std::dec;
}

void HGCalNeighbourVerify::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("nameDetector", "HGCalHESiliconSensitive");
  desc.add<int>("waferU", 2);
  desc.add<int>("waferV", 0);
  desc.add<int>("cellU", 10);
  desc.add<int>("cellV", 0);
  descriptions.add("hgcalNeighbourVerify", desc);
}

// ------------ method called to produce the data  ------------
void HGCalNeighbourVerify::beginRun(edm::Run const &iRun, edm::EventSetup const &iSetup) {
  //initiating hgc Geometry
  edm::LogVerbatim("HGCalGeom") << "Tries to initialize HGCalGeometry and HGCalDDDConstants for " << nameDetector_;
  const edm::ESHandle<HGCalGeometry> &hgcGeom = iSetup.getHandle(tok_hgcal_);
  if (hgcGeom.isValid()) {
    const HGCalGeometry *geom = hgcGeom.product();
    edm::LogVerbatim("HGCalGeom") << "Loaded HGCalDDConstants for " << nameDetector_;
    std::vector<DetId> detIdx = geom->getValidDetIds(dets_);
    edm::LogVerbatim("HGCalGeom") << "Gets " << detIdx.size() << " valid ID's for detector " << dets_;
    std::vector<DetId> detIds;
    static constexpr uint32_t mask1 = 0xFFFFF;
    for (unsigned int k = 0; k < detIdx.size(); ++k) {
      if (((detIdx[k].rawId())&mask1) == idUV_)
	detIds.emplace_back(detIdx[k]);
    }
    edm::LogVerbatim("HGCalGeom") << "Gets " << detIds.size() << " valid ID's for detector " << dets_;
    for (unsigned int k = 0; k < detIds.size(); ++k)
      edm::LogVerbatim("HGCalGeom") << " [" << k << "] " << HGCSiliconDetId(detIds[k]);
    std::unique_ptr<HGCalNeighbourFinder> finder = std::make_unique<HGCalNeighbourFinder>(geom->topology().dddConstants());
    for (unsigned int k = 0; k < detIds.size(); ++k) {
      HGCSiliconDetId id(detIds[k]);
      std::vector<uint32_t> ids = finder->nearestNeighboursOfDetId(id.rawId());
      unsigned int nn(0);
      for (auto const &idZ : ids)
	if (idZ != 0)
	  ++nn;
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << id.detType() << " Type " << id.waferTypeX() << " z " << id.zside() << " Layer " << id.layer() << " Wafer " << id.waferU() << ":" << id.waferV() << " Cell " << id.cellU() << ":" << id.cellV() << " has " << nn << " neighbours:";
      unsigned int k1(0);
      for (auto const &idZ : ids) {
	if (idZ != 0) {
	  HGCSiliconDetId idx(idZ);
	  edm::LogVerbatim("HGCalGeom") << "[" << k1 << "] Layer " << idx.layer() << " Wafer " << idx.waferU() << ":" << idx.waferV() << " Cell " << idx.cellU() << ":" << idx.cellV();
	  ++k1;
	}
      }
    }
  } else {
    edm::LogWarning("HGCalGeom") << "Cannot initiate HGCalGeometry for " << nameDetector_ << std::endl;
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalNeighbourVerify);
