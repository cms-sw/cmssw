#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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

class HGCalGeometryNewCornersTest : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit HGCalGeometryNewCornersTest(const edm::ParameterSet&);
  ~HGCalGeometryNewCornersTest() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override {}
  void endRun(edm::Run const&, edm::EventSetup const&) override {}

private:
  const std::string nameDetector_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> geomToken_;
  const std::vector<int> layers_;
  const std::vector<int> waferU_, waferV_, types_;
  const bool debug_;
};

HGCalGeometryNewCornersTest::HGCalGeometryNewCornersTest(const edm::ParameterSet& iC)
    : nameDetector_(iC.getParameter<std::string>("detectorName")),
      geomToken_(esConsumes<HGCalGeometry, IdealGeometryRecord, edm::Transition::BeginRun>(
          edm::ESInputTag{"", nameDetector_})),
      layers_(iC.getParameter<std::vector<int>>("layers")),
      waferU_(iC.getParameter<std::vector<int>>("waferUs")),
      waferV_(iC.getParameter<std::vector<int>>("waferVs")),
      types_(iC.getParameter<std::vector<int>>("types")),
      debug_(iC.getParameter<bool>("debugFlag")) {}

void HGCalGeometryNewCornersTest::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<int> layer = {27, 28, 29, 30, 31, 32};
  std::vector<int> waferU = {-2, -3, 1, -8, 2, 8};
  std::vector<int> waferV = {0, -2, 3, 0, 9, 0};
  std::vector<int> type = {0, 0, 0, 2, 2, 2};
  desc.add<std::string>("detectorName", "HGCalHESiliconSensitive");
  desc.add<std::vector<int>>("layers", layer);
  desc.add<std::vector<int>>("waferUs", waferU);
  desc.add<std::vector<int>>("waferVs", waferV);
  desc.add<std::vector<int>>("types", type);
  desc.add<bool>("debugFlag", false);
  descriptions.add("hgcalGeometryNewCornersTest", desc);
}

void HGCalGeometryNewCornersTest::beginRun(const edm::Run&, const edm::EventSetup& iSetup) {
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
      HGCSiliconDetId detId(det, 1, types_[k], lay - layerOff, waferU_[k], waferV_[k], 0, 0);
      GlobalPoint global = geom->getPosition(DetId(detId), debug_);
      double phi2 = global.phi();
      auto xy = geom->topology().dddConstants().waferPosition(lay - layerOff, waferU_[k], waferV_[k], true, debug_);
      double phi1 = std::atan2(xy.second, xy.first);
      edm::LogVerbatim("HGCalGeom") << "Layer: " << lay << " U " << waferU_[k] << " V " << waferV_[k] << " Position ("
                                    << xy.first << ", " << xy.second << ") phi " << convertRadToDeg(phi1);
      edm::LogVerbatim("HGCalGeom") << detId << " Position " << global << " phi " << convertRadToDeg(phi2);
      std::vector<GlobalPoint> corners = geom->getNewCorners(DetId(detId), debug_);
      std::ostringstream st1;
      for (auto const& it : corners)
        st1 << it << ", ";
      edm::LogVerbatim("HGCalGeom") << "Corners: " << st1.str();
    }
  }
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HGCalGeometryNewCornersTest);
