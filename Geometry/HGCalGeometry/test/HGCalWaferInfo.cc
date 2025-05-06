// -*- C++ -*-
//
// Package:    HGCalWaferInfo
// Class:      HGCalWaferInfo
//
/**\class HGCalWaferInfo HGCalWaferInfo.cc
 test/HGCalWaferInfo.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Sunanda Banerjee
//         Created:  Mon 2025/05/02
//
//

// system include files
#include <fstream>
#include <iostream>
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

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HGCalWaferInfo : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalWaferInfo(const edm::ParameterSet&);
  ~HGCalWaferInfo() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const std::string name_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> geomToken_;
};

HGCalWaferInfo::HGCalWaferInfo(const edm::ParameterSet& iC)
    : name_(iC.getParameter<std::string>("detector")),
      geomToken_(esConsumes<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag("", name_))) {}

void HGCalWaferInfo::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("detector", "HGCalEESensitive");
  descriptions.add("hgcalWaferInfoEE", desc);
}

// ------------ method called to produce the data  ------------
void HGCalWaferInfo::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& geom = &iSetup.getData(geomToken_);
  DetId::Detector det = (name_ == "HGCalHESiliconSensitive") ? DetId::HGCalHSi : DetId::HGCalEE;
  edm::LogVerbatim("HGCalGeom") << "Perform test for " << name_ << " Detector " << det;

  const std::vector<DetId>& ids = geom->getValidDetIds();
  edm::LogVerbatim("HGCalGeom") << "Use " << ids.size() << " valid ids for " << geom->cellElement();
  std::string parts[26] = {"Full",  "Five", "ChopTwo", "ChopTwoM", "Half",     "Semi",    "Semi2",   "Three",   "Half2",
                           "Five2", "????", "LDTop",   "LDBottom", "LDLeft",   "LDRight", "LDFive",  "LDThree", "????",
                           "????",  "????", "????",    "HDTop",    "HDBottom", "HDLeft",  "HDRight", "HDFive"};
  std::string types[4] = {"HD120", "LD200", "LD300", "HD200"};

  for (auto const& id : ids) {
    if ((id.det() == DetId::HGCalEE) || (id.det() == DetId::HGCalHSi)) {
      HGCSiliconDetId detId(id);
      HGCalParameters::waferInfo info =
          geom->topology().dddConstants().waferInfo(detId.layer(), detId.waferU(), detId.waferV());
      edm::LogVerbatim("HGCalGeom") << "ID: " << detId << " Type " << info.type << ":" << types[info.type] << " Part "
                                    << info.part << ":" << parts[info.part] << " Orient " << info.orient << " placement "
				    << geom->topology().dddConstants().placementIndex(detId) << " Cassette "
                                    << info.cassette << " at " << geom->getPosition(id, true, false);
    } else {
      edm::LogVerbatim("HGCalGeom") << "Illegal Det " << id.det() << " in " << std::hex << id.rawId() << std::dec
                                    << " ERROR";
    }
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalWaferInfo);
