#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

#include <vector>

class HGCalTestRecHitTools : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalTestRecHitTools(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
  hgcal::RecHitTools tool_;
};

HGCalTestRecHitTools::HGCalTestRecHitTools(const edm::ParameterSet&)
    : tok_geom_(esConsumes<CaloGeometry, CaloGeometryRecord>()) {}

void HGCalTestRecHitTools::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("hgcalRecHitTools", desc);
}

void HGCalTestRecHitTools::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {
  const CaloGeometry geo = iSetup.getData(tok_geom_);
  tool_.setGeometry(geo);

  edm::LogVerbatim("HGCalGeom") << "EE: Last Layer " << tool_.lastLayerEE();
  edm::LogVerbatim("HGCalGeom") << "FH: Last Layer " << tool_.lastLayerFH();
  edm::LogVerbatim("HGCalGeom") << "BH: First & Last Layer " << tool_.firstLayerBH() << ":" << tool_.lastLayerBH();
  edm::LogVerbatim("HGCalGeom") << "Last Layer " << tool_.lastLayer();
  std::vector<DetId::Detector> dets = {DetId::HGCalEE, DetId::HGCalHSi, DetId::HGCalHSc};
  for (const auto& det : dets) {
    auto layer = tool_.firstAndLastLayer(det, 0);
    edm::LogVerbatim("HGCalGeom") << "First & Last Layer for Det " << det << " are " << layer.first << ":"
                                  << layer.second;
  }

  edm::LogVerbatim("HGCalGeom") << "Maximum # of wafers per layer " << tool_.maxNumberOfWafersPerLayer();
  edm::LogVerbatim("HGCalGeom") << "Maximum # of iphi: " << tool_.getScintMaxIphi();
  edm::LogVerbatim("HGCalGeom") << "Geometry type " << tool_.getGeometryType();
}

DEFINE_FWK_MODULE(HGCalTestRecHitTools);
