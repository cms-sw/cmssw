// system include files
#include <memory>
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
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferMask.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

class HGCalPartialWaferTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalPartialWaferTester(const edm::ParameterSet&);
  ~HGCalPartialWaferTester() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const std::string nameSense_;
  const int orientation_, partialType_, nTrials_;
  const edm::ESGetToken<HGCalDDDConstants, IdealGeometryRecord> dddToken_;
};

HGCalPartialWaferTester::HGCalPartialWaferTester(const edm::ParameterSet& iC)
    : nameSense_(iC.getParameter<std::string>("NameSense")),
      orientation_(iC.getParameter<int>("waferOrientation")),
      partialType_(iC.getParameter<int>("partialType")),
      nTrials_(iC.getParameter<int>("numberOfTrials")),
      dddToken_(esConsumes<HGCalDDDConstants, IdealGeometryRecord>(edm::ESInputTag{"", nameSense_})) {
  edm::LogVerbatim("HGCalGeom") << "Test positions for partial wafer type " << partialType_ << " Orientation "
                                << orientation_ << " for " << nameSense_;
}

void HGCalPartialWaferTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("NameSense", "HGCalHESiliconSensitive");
  desc.add<int>("waferOrientation", 0);
  desc.add<int>("partialType", 11);
  desc.add<int>("numberOfTrials", 1000);
  descriptions.add("hgcalPartialWaferTester", desc);
}

void HGCalPartialWaferTester::analyze(const edm::Event&, const edm::EventSetup& iSetup) {
  const HGCalDDDConstants& hgdc = iSetup.getData(dddToken_);
  int indx(-1), type(-1);
  for (auto itr = hgdc.getParameter()->waferInfoMap_.begin(); itr != hgdc.getParameter()->waferInfoMap_.end(); ++itr) {
    if (((itr->second).part == partialType_) && ((itr->second).orient == orientation_)) {
      indx = itr->first;
      type = (itr->second).type;
      break;
    }
  }

  if (indx > 0) {
    int all(0), error(0);
    int layer = HGCalWaferIndex::waferLayer(indx);
    int waferU = HGCalWaferIndex::waferU(indx);
    int waferV = HGCalWaferIndex::waferV(indx);
    auto xy = hgdc.waferPosition(layer, waferU, waferV, true, false);
    edm::LogVerbatim("HGCalGeom") << "Wafer " << waferU << ":" << waferV << " in layer " << layer << " at " << xy.first
                                  << ":" << xy.second << "\n\n";
    int nCells = (type == 0) ? HGCSiliconDetId::HGCalFineN : HGCSiliconDetId::HGCalCoarseN;
    for (int i = 0; i < nTrials_; i++) {
      int ui = (2 * nCells * (float)rand() / RAND_MAX);
      int vi = (2 * nCells * (float)rand() / RAND_MAX);
      if ((ui < 2 * nCells) && (vi < 2 * nCells) && ((vi - ui) < nCells) && ((ui - vi) <= nCells) &&
          HGCalWaferMask::goodCell(ui, vi, partialType_)) {
        ++all;
        auto xy = hgdc.locateCell(layer, waferU, waferV, ui, vi, true, true, false, false);
        int lay(layer), cU(0), cV(0), wType(-1), wU(0), wV(0);
        double wt(0);
        hgdc.waferFromPosition(HGCalParameters::k_ScaleToDDD * xy.first,
                               HGCalParameters::k_ScaleToDDD * xy.second,
                               lay,
                               wU,
                               wV,
                               cU,
                               cV,
                               wType,
                               wt,
                               false,
                               true);
        std::string comment =
            ((wType == type) && (layer == lay) && (waferU == wU) && (waferV == wV) && (ui == cU) && (vi == cV))
                ? ""
                : " ***** ERROR *****";
        edm::LogVerbatim("HGCalGeom") << "Layer " << layer << ":" << lay << " waferU " << waferU << ":" << wU
                                      << " waferV " << waferV << ":" << wV << " Type " << type << ":" << wType
                                      << " cellU " << ui << ":" << cU << " cellV " << vi << ":" << cV << " position "
                                      << xy.first << ":" << xy.second << comment;
        if (comment != "")
          ++error;
      }
    }
    edm::LogVerbatim("HGCalGeom") << "\n\nFound " << error << " errors among " << all << ":" << nTrials_ << " trials";
  } else {
    edm::LogVerbatim("HGCalGeom") << "\n\nCannot find a wafer of type " << partialType_ << " and orientation "
                                  << orientation_ << " for " << nameSense_;
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalPartialWaferTester);
