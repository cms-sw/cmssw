#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/EZArrayFL.h"

#include <iomanip>
#include <iostream>

typedef EZArrayFL<GlobalPoint> CornersVec;

class EcalPreshowerCellParameterDump : public edm::one::EDAnalyzer<> {
public:
  explicit EcalPreshowerCellParameterDump(const edm::ParameterSet&) {}

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
};

void EcalPreshowerCellParameterDump::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {
  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  const CaloGeometry* geo = pG.product();
  const CaloSubdetectorGeometry* ecalGeom =
      static_cast<const CaloSubdetectorGeometry*>(geo->getSubdetectorGeometry(DetId::Ecal, EcalPreshower));

  std::cout << "\n\nStudy Detector = Ecal SubDetector = ES"
            << "\n======================================\n\n";
  const std::vector<DetId>& ids = ecalGeom->getValidDetIds(DetId::Ecal, EcalPreshower);
  int nall(0);
  for (auto id : ids) {
    ++nall;
    auto geom = ecalGeom->getGeometry(id);
    ESDetId esid(id);

    std::cout << "Cell[" << nall << "] " << esid << " geom->getPosition " << std::setprecision(4) << geom->getPosition()
              << " BackPoint " << geom->getBackPoint() << " [rho,eta:etaSpan,phi:phiSpan] (" << geom->rhoPos() << ", "
              << geom->etaPos() << ":" << geom->etaSpan() << ", " << geom->phiPos() << ":" << geom->phiSpan() << ")";

    const CaloCellGeometry::CornersVec& corners(geom->getCorners());

    for (unsigned int ci(0); ci != corners.size(); ci++) {
      std::cout << " Corner: " << ci << "  Location" << corners[ci] << " ; ";
    }

    std::cout << std::endl;
  }
  std::cout << "\n\nDumps a total of : " << nall << " cells of the detector\n" << std::endl;
}

DEFINE_FWK_MODULE(EcalPreshowerCellParameterDump);
