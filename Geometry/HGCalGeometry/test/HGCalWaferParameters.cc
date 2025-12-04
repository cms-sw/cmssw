// -*- C++ -*-
//
// Package:    HGCalWaferParameters
// Class:      HGCalWaferParameters
//
/**\class HGCalWaferParameters HGCalWaferParameters.cc
 test/HGCalWaferParameters.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2025/12/04
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
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomUtils.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HGCalWaferParameters : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit HGCalWaferParameters(const edm::ParameterSet&);
  ~HGCalWaferParameters() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginJob() override {}
  void beginRun(edm::Run const &, edm::EventSetup const &) override;
  void analyze(edm::Event const &iEvent, edm::EventSetup const &) override {}
  void endRun(edm::Run const &, edm::EventSetup const &) override {}
  void endJob() override {}

private:
  const std::string nameDetector_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> tok_hgcal_;
  std::vector<HGCSiliconDetId> detIds_;
};

HGCalWaferParameters::HGCalWaferParameters(const edm::ParameterSet& iC)
    : nameDetector_(iC.getParameter<std::string>("nameDetector")),
      tok_hgcal_{esConsumes<HGCalGeometry, IdealGeometryRecord, edm::Transition::BeginRun>(edm::ESInputTag{"", nameDetector_})} {
  edm::LogVerbatim("HGCGeom") << "Test HGCSilicon DetID for " << nameDetector_;

}

void HGCalWaferParameters::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("nameDetector", "HGCalHESiliconSensitive");
  descriptions.add("hgcalWaferParameters", desc);
}

// ------------ method called to produce the data  ------------
void HGCalWaferParameters::beginRun(edm::Run const &iRun, edm::EventSetup const &iSetup) {
  edm::LogVerbatim("HGCGeom") << "Try to initialize HGCalGeometry and HGCalDDDConstants for " << nameDetector_;
  const edm::ESHandle<HGCalGeometry> &hgcGeom = iSetup.getHandle(tok_hgcal_);
  if (hgcGeom.isValid()) {
    const HGCalGeometry *geom = hgcGeom.product();
    edm::LogVerbatim("HGCGeom") << "Loaded HGCalDDConstants for " << nameDetector_;

    const std::vector<DetId>& ids = geom->getValidDetIds();
    for (unsigned int k = 0; k < ids.size(); ++k) {
      std::ostringstream st1;
      HGCSiliconDetId detId(ids[k]);
      st1 << "Hit[" << k << "] " << detId;
      int zside = detId.zside();
      int layer = detId.layer();
      int waferU = detId.waferU();
      int waferV = detId.waferV();
      st1 << " Exist:HD?:Patial?:Placement " << geom->topology().dddConstants().waferExist(layer, waferU, waferV) << ":" << geom->topology().dddConstants().waferIsHD(layer, waferU, waferV) << ":" << geom->topology().dddConstants().waferPartial(layer, waferU, waferV) << ":" << geom->topology().dddConstants().waferPlacementIndex(zside, layer, waferU, waferV);
      edm::LogVerbatim("HGCGeom") << st1.str();
    }
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalWaferParameters);
