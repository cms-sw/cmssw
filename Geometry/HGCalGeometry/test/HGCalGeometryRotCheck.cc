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
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"

class HGCalGeometryRotCheck : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit HGCalGeometryRotCheck(const edm::ParameterSet&);
  ~HGCalGeometryRotCheck() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override {}
  void endRun(edm::Run const&, edm::EventSetup const&) override {}

private:
  const std::string nameDetector_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> geomToken_;
  const std::vector<int> layers_;
};

HGCalGeometryRotCheck::HGCalGeometryRotCheck(const edm::ParameterSet& iC)
    : nameDetector_(iC.getParameter<std::string>("detectorName")),
      geomToken_(esConsumes<HGCalGeometry, IdealGeometryRecord, edm::Transition::BeginRun>(
          edm::ESInputTag{"", nameDetector_})),
      layers_(iC.getParameter<std::vector<int>>("layers")) {}

void HGCalGeometryRotCheck::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<int> layer = {27, 28, 29, 30, 31, 32, 33};
  desc.add<std::string>("detectorName", "HGCalHESiliconSensitive");
  desc.add<std::vector<int>>("layers", layer);
  descriptions.add("hgcalGeometryRotCheck", desc);
}

void HGCalGeometryRotCheck::beginRun(const edm::Run&, const edm::EventSetup& iSetup) {
  const auto& geomR = iSetup.getData(geomToken_);
  const HGCalGeometry* geom = &geomR;
  int layerF = *(layers_.begin());
  int layerL = *(--layers_.end());
  int layerOff = geom->topology().dddConstants().getLayerOffset();
  edm::LogVerbatim("HGCalGeom") << nameDetector_ << " with layers in the range " << layerF << ":" << layerL
                                << " Offset " << layerOff;

  auto rrF = geom->topology().dddConstants().rangeRLayer(layerF - layerOff, true);
  auto rrE = geom->topology().dddConstants().rangeRLayer(layerL - layerOff, true);
  edm::LogVerbatim("HGCalGeom") << " RFront " << rrF.first << ":" << rrF.second << " RBack " << rrE.first << ":"
                                << rrE.second;
  double r = rrE.first + 5.0;
  const int nPhi = 10;
  while (r <= rrF.second) {
    for (int k = 0; k < nPhi; ++k) {
      double phi = 2.0 * k * M_PI / nPhi;
      for (auto lay : layers_) {
        double zz = geom->topology().dddConstants().waferZ(lay - layerOff, true);
        GlobalPoint global1(r * cos(phi), r * sin(phi), zz);
        DetId id = geom->getClosestCellHex(global1, true);
        HGCSiliconDetId detId = HGCSiliconDetId(id);
        GlobalPoint global2 = geom->getPosition(id);
        double dx = global1.x() - global2.x();
        double dy = global1.y() - global2.y();
        double dR = std::sqrt(dx * dx + dy * dy);
        edm::LogVerbatim("HGCalGeom") << "Layer: " << lay << " ID " << detId << " I/P " << global1 << " O/P " << global2
                                      << " dR " << dR;
      }
    }
    r += 100.0;
  }
}

//define this as a plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HGCalGeometryRotCheck);
