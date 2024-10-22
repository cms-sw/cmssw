#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "DataFormats/Math/interface/angle_units.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"

using namespace angle_units::operators;

class HGCalGeometryRotTest : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit HGCalGeometryRotTest(const edm::ParameterSet&);
  ~HGCalGeometryRotTest() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override {}
  void endRun(edm::Run const&, edm::EventSetup const&) override {}

private:
  const std::string nameDetector_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> geomToken_;
  const std::vector<int> layers_;
  const std::vector<int> waferU_, waferV_, cellU_, cellV_, types_;
};

HGCalGeometryRotTest::HGCalGeometryRotTest(const edm::ParameterSet& iC)
    : nameDetector_(iC.getParameter<std::string>("detectorName")),
      geomToken_(esConsumes<HGCalGeometry, IdealGeometryRecord, edm::Transition::BeginRun>(
          edm::ESInputTag{"", nameDetector_})),
      layers_(iC.getParameter<std::vector<int>>("layers")),
      waferU_(iC.getParameter<std::vector<int>>("waferUs")),
      waferV_(iC.getParameter<std::vector<int>>("waferVs")),
      cellU_(iC.getParameter<std::vector<int>>("cellUs")),
      cellV_(iC.getParameter<std::vector<int>>("cellVs")),
      types_(iC.getParameter<std::vector<int>>("types")) {}

void HGCalGeometryRotTest::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<int> layer = {27, 28, 29, 30, 31, 32};
  std::vector<int> waferU = {-2, -3, 1, -8, 2, 8};
  std::vector<int> waferV = {0, -2, 3, 0, 9, 0};
  std::vector<int> cellU = {16, 4, 8, 11, 11, 5};
  std::vector<int> cellV = {20, 10, 17, 13, 9, 2};
  std::vector<int> type = {0, 0, 0, 2, 2, 2};
  desc.add<std::string>("detectorName", "HGCalHESiliconSensitive");
  desc.add<std::vector<int>>("layers", layer);
  desc.add<std::vector<int>>("waferUs", waferU);
  desc.add<std::vector<int>>("waferVs", waferV);
  desc.add<std::vector<int>>("cellUs", cellU);
  desc.add<std::vector<int>>("cellVs", cellV);
  desc.add<std::vector<int>>("types", type);
  descriptions.add("hgcalGeometryRotTest", desc);
}

void HGCalGeometryRotTest::beginRun(const edm::Run&, const edm::EventSetup& iSetup) {
  const auto& geomR = iSetup.getData(geomToken_);
  const HGCalGeometry* geom = &geomR;
  DetId::Detector det = (nameDetector_ == "HGCalEESensitive") ? DetId::HGCalEE : DetId::HGCalHSi;
  int layerF = *(layers_.begin());
  int layerL = *(--layers_.end());
  int layerOff = geom->topology().dddConstants().getLayerOffset();
  edm::LogVerbatim("HGCalGeom") << nameDetector_ << " with layers in the range " << layerF << ":" << layerL
                                << " Offset " << layerOff << " and for " << waferU_.size() << " wafers and cells";

  for (unsigned int k = 0; k < waferU_.size(); ++k) {
    for (auto lay : layers_) {
      HGCSiliconDetId detId(det, 1, types_[k], lay - layerOff, waferU_[k], waferV_[k], cellU_[k], cellV_[k]);
      GlobalPoint global = geom->getPosition(DetId(detId));
      double phi2 = global.phi();
      auto xy = geom->topology().dddConstants().waferPosition(lay - layerOff, waferU_[k], waferV_[k], true, false);
      double phi1 = std::atan2(xy.second, xy.first);
      edm::LogVerbatim("HGCalGeom") << "Layer: " << lay << " U " << waferU_[k] << " V " << waferV_[k] << " Position ("
                                    << xy.first << ", " << xy.second << ") phi " << convertRadToDeg(phi1);
      edm::LogVerbatim("HGCalGeom") << detId << " Position " << global << " phi " << convertRadToDeg(phi2);
    }
  }
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HGCalGeometryRotTest);
