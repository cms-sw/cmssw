// -*- C++ -*-
//
// Package:    HGCalListValidCells
// Class:      HGCalListValidCells
//
/**\class HGCalListValidCells HGCalListValidCells.cc
 test/HGCalListValidCells.cc

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
#include <algorithm>
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

class HGCalListValidCells : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalListValidCells(const edm::ParameterSet&);
  ~HGCalListValidCells() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const std::string name_;
  const int partialType_, verbosity_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> geomToken_;
};

HGCalListValidCells::HGCalListValidCells(const edm::ParameterSet& iC)
    : name_(iC.getParameter<std::string>("detector")),
      partialType_(iC.getParameter<int>("partialType")),
      verbosity_(iC.getParameter<int>("verbosity")),
      geomToken_(esConsumes<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag("", name_))) {}

void HGCalListValidCells::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("detector", "HGCalEESensitive");
  desc.add<int>("partialType", 16);
  desc.add<int>("verbosity", 0);
  descriptions.add("hgcalListValidCellsEE", desc);
}

// ------------ method called to produce the data  ------------
void HGCalListValidCells::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const auto& geom = &iSetup.getData(geomToken_);
  DetId::Detector det = (name_ == "HGCalHESiliconSensitive") ? DetId::HGCalHSi : DetId::HGCalEE;
  edm::LogVerbatim("HGCalGeom") << "Perform test for " << name_ << " Detector " << det;

  std::string parts[26] = {"Full",  "Five", "ChopTwo", "ChopTwoM", "Half",     "Semi",    "Semi2",   "Three",   "Half2",
                           "Five2", "????", "LDTop",   "LDBottom", "LDLeft",   "LDRight", "LDFive",  "LDThree", "????",
                           "????",  "????", "????",    "HDTop",    "HDBottom", "HDLeft",  "HDRight", "HDFive"};
  const std::vector<DetId>& ids = geom->getValidDetIds();
  edm::LogVerbatim("HGCalGeom") << "Find the list of valid DetIds of type " << partialType_ << ":" << parts[partialType_] << " among a list of " << ids.size() << " valid ids of " << geom->cellElement();
  std::vector<HGCSiliconDetId> detIds;

  for (auto const& id : ids) {
    if ((id.det() == DetId::HGCalEE) || (id.det() == DetId::HGCalHSi)) {
      HGCSiliconDetId detId(id);
      HGCalParameters::waferInfo info =
          geom->topology().dddConstants().waferInfo(detId.layer(), detId.waferU(), detId.waferV());
      if (info.part == partialType_) {
	if (std::find(detIds.begin(), detIds.end(), detId) == detIds.end())
	  detIds.emplace_back(detId);
      }
    } else {
      edm::LogVerbatim("HGCalGeom") << "Illegal Det " << id.det() << " in " << std::hex << id.rawId() << std::dec
                                    << " ERROR";
    }
  }
  edm::LogVerbatim("HGCalGeom") << "There are " << detIds.size() << " valid Ids with partial type " << partialType_ << ":" << parts[partialType_];
  if (verbosity_ > 0) {
    for (auto const detId : detIds)
      edm::LogVerbatim("HGCalGeom") << " " << detId;
  }
    
  if (detIds.size() > 0) {
    std::vector<int> cellPatterns;
    for (auto const& detId : detIds) {
      int iuv = (100 * detId.cellU() + detId.cellV());
      if (std::find(cellPatterns.begin(), cellPatterns.end(), iuv) == cellPatterns.end())
	cellPatterns.emplace_back(iuv);
    }
    std::sort(cellPatterns.begin(), cellPatterns.end());
    edm::LogVerbatim("HGCalGeom") << "There are " << cellPatterns.size() << " different cell patterns:";
    for (const auto& iuv : cellPatterns) {
      int u = ((iuv / 100) % 100);
      int v = (iuv % 100);
      edm::LogVerbatim("HGCalGeom") << "u = " << std::setw(3) << u << " v = " << std::setw(3) << v;
    }
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalListValidCells);
