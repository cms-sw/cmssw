#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include <iostream>
#include <string>

class CaloCellGeometryTester : public edm::one::EDAnalyzer<> {
public:
  explicit CaloCellGeometryTester(const edm::ParameterSet&);
  ~CaloCellGeometryTester(void) override = default;

  void analyze(edm::Event const&, edm::EventSetup const&) override;

private:
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloToken_;
};

CaloCellGeometryTester::CaloCellGeometryTester(const edm::ParameterSet&)
    : caloToken_{esConsumes<CaloGeometry, CaloGeometryRecord>(edm::ESInputTag{})} {}

void CaloCellGeometryTester::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {
  // get handles to calogeometry and calotopology
  const CaloGeometry* geo = &iSetup.getData(caloToken_);

  const int ncalo = 11;
  int dets[ncalo] = {3, 3, 3, 4, 4, 4, 4, 8, 9, 10, 6};
  int subd[ncalo] = {1, 2, 3, 1, 2, 4, 3, 0, 0, 0, 6};
  std::string name[ncalo] = {"EB", "EE", "ES", "HB", "HE", "HF", "HO", "HGCEE", "HGCHESil", "HGCHESci", "HFNose"};
  for (unsigned int k = 0; k < ncalo; ++k) {
    const CaloSubdetectorGeometry* geom = geo->getSubdetectorGeometry((DetId::Detector)(dets[k]), subd[k]);
    if (geom) {
      edm::LogVerbatim("HGCalGeomX") << name[k] << " has "
                                     << geom->getValidDetIds((DetId::Detector)(dets[k]), subd[k]).size()
                                     << " valid cells";
      if (k >= 7)
        edm::LogVerbatim("HGCalGeomX") << "Number of valid GeomID "
                                       << ((HGCalGeometry*)(geom))->getValidGeomDetIds().size();
    } else {
      edm::LogVerbatim("HGCalGeomX") << name[k] << " is not present in the current scenario";
    }
  }
}

DEFINE_FWK_MODULE(CaloCellGeometryTester);
