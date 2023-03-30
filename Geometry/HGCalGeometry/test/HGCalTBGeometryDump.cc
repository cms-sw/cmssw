#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/HGCalTBGeometry.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include <iostream>

class HGCalTBGeometryDump : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalTBGeometryDump(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const std::vector<std::string> names_;
  std::vector<edm::ESGetToken<HGCalTBGeometry, IdealGeometryRecord>> geomTokens_;
};

HGCalTBGeometryDump::HGCalTBGeometryDump(const edm::ParameterSet& iC)
    : names_(iC.getParameter<std::vector<std::string>>("detectorNames")) {
  for (unsigned int k = 0; k < names_.size(); ++k) {
    edm::LogVerbatim("HGCalGeomX") << "Study detector [" << k << "] " << names_[k] << std::endl;
    geomTokens_.emplace_back(esConsumes<HGCalTBGeometry, IdealGeometryRecord>(edm::ESInputTag{"", names_[k]}));
  }
}

void HGCalTBGeometryDump::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<std::string> names = {"HGCalEESensitive", "HGCalHESiliconSensitive"};
  desc.add<std::vector<std::string>>("detectorNames", names);
  descriptions.add("hgcalTBGeometryDump", desc);
}

void HGCalTBGeometryDump::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {
  for (unsigned int k = 0; k < names_.size(); ++k) {
    const auto& geomR = iSetup.getData(geomTokens_[k]);
    const HGCalTBGeometry* geom = &geomR;
    const std::vector<DetId>& ids = geom->getValidDetIds();
    edm::LogVerbatim("HGCalGeomX") << ids.size() << " valid Ids for detector " << names_[k];
    int nall(0);
    for (auto id : ids) {
      ++nall;
      auto cell = geom->getGeometry(id);
      HGCalDetId hid(id);
      edm::LogVerbatim("HGCalGeomX") << "[" << nall << "] " << hid << " Reference " << std::setprecision(4)
                                     << cell->getPosition() << " Back " << cell->getBackPoint() << " [r,eta,phi] ("
                                     << cell->rhoPos() << ", " << cell->etaPos() << ":" << cell->etaSpan() << ", "
                                     << cell->phiPos() << ":" << cell->phiSpan() << ")";
    }
    edm::LogVerbatim("HGCalGeomX") << "\n\nDumps " << nall << " cells of the detector\n";
  }
}

DEFINE_FWK_MODULE(HGCalTBGeometryDump);
