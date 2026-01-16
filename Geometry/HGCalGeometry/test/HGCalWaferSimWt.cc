#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferType.h"
#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include <iostream>

class HGCalWaferSimWt : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalWaferSimWt(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const std::vector<std::string> names_;
  const bool debug_;
  std::vector<edm::ESGetToken<HGCalGeometry, IdealGeometryRecord>> geomTokens_;
};

HGCalWaferSimWt::HGCalWaferSimWt(const edm::ParameterSet& iC)
    : names_(iC.getParameter<std::vector<std::string>>("detectorNames")), debug_(iC.getParameter<bool>("debug")) {
  for (unsigned int k = 0; k < names_.size(); ++k) {
    edm::LogVerbatim("HGCalGeomX") << "Study detector [" << k << "] " << names_[k] << std::endl;
    geomTokens_.emplace_back(esConsumes<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag{"", names_[k]}));
  }
}

void HGCalWaferSimWt::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<std::string> names = {"HGCalEESensitive", "HGCalHESiliconSensitive"};
  desc.add<std::vector<std::string>>("detectorNames", names);
  desc.add<bool>("debug", false);
  descriptions.add("hgcalWaferSimWt", desc);
}

void HGCalWaferSimWt::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {
  std::vector<std::string> parType = {"Full",    "Five",     "ChopTwo", "ChopTwoM", "Half",  "Semi",     "Semi2",
                                      "Three",   "Half2",    "Five2",   "JK10",     "LDTop", "LDBottom", "LDLeft",
                                      "LDRight", "LDFive",   "LDThree", "JK17",     "JK18",  "JK19",     "JK20",
                                      "HDTop",   "HDBottom", "HDLeft",  "HDRight",  "HDFive"};
  std::vector<std::string> detType = {"HD120", "LD200", "LD300", "HD200"};
  for (unsigned int k = 0; k < names_.size(); ++k) {
    const auto& geomR = iSetup.getData(geomTokens_[k]);
    const HGCalGeometry* geom = &geomR;
    const std::vector<DetId>& ids = geom->getValidDetIds();
    edm::LogVerbatim("HGCalGeomX") << ids.size() << " valid Ids for detector " << names_[k] << std::endl;
    int nall(0), ntypes(0);
    std::vector<int> idxs;
    for (auto id : ids) {
      ++nall;
      HGCSiliconDetId hid(id);
      auto cell = geom->getPosition(hid, false);
      int type = hid.type();
      int waferU = hid.waferU();
      int part = geom->topology().dddConstants().partialWaferType(hid.layer(), waferU, hid.waferV());
      int idx = part * 10 + type;
      if (std::find(idxs.begin(), idxs.end(), idx) == idxs.end()) {
        ++ntypes;
        idxs.push_back(idx);
        double xpos = 10.0 * (cell.x());
        double ypos = 10.0 * (cell.y());
        int waferU, waferV, cellU, cellV, cellType;
        double wt;
        if (debug_)
          edm::LogVerbatim("HGCalGeomX") << hid << " at (" << xpos << ", " << ypos << ")";
        geom->topology().dddConstants().waferFromPosition(
            xpos, ypos, hid.zside(), hid.layer(), waferU, waferV, cellU, cellV, cellType, wt, false, true);
        std::string stype = (type >= 0 && type <= 3) ? detType[type] : ("JK" + std::to_string(type));
        std::string spart = (part >= 0 && part <= 25) ? parType[part] : ("JK" + std::to_string(part));
        int index = HGCalWaferIndex::waferIndex(hid.layer(), waferU, waferV);
        int celltypeX = HGCalWaferType::getType(index, geom->topology().dddConstants().getParameter()->waferInfoMap_);
        edm::LogVerbatim("HGCalGeomX") << "[" << ntypes << "] " << stype << " " << spart << " wt " << wt << " for "
                                       << hid << " at " << xpos << ":" << ypos << " Wafer " << waferU << ":" << waferV
                                       << " cell " << cellU << ":" << cellV << " Type " << cellType << ":" << celltypeX;
      }
    }
    edm::LogVerbatim("HGCalGeomX") << "\n\nFinds " << idxs.size() << " different wafer types among " << nall
                                   << " cells of the detector\n";
  }
}

DEFINE_FWK_MODULE(HGCalWaferSimWt);
