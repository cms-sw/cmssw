#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
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

class EcalBarrelCellParameterDump : public edm::one::EDAnalyzer<> {
public:
  explicit EcalBarrelCellParameterDump(const edm::ParameterSet&);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
};

EcalBarrelCellParameterDump::EcalBarrelCellParameterDump(const edm::ParameterSet&)
    : tok_geom_(esConsumes<CaloGeometry, CaloGeometryRecord>()) {}

void EcalBarrelCellParameterDump::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {
  const CaloGeometry* geo = &iSetup.getData(tok_geom_);
  const CaloSubdetectorGeometry* ecalGeom =
      static_cast<const CaloSubdetectorGeometry*>(geo->getSubdetectorGeometry(DetId::Ecal, EcalBarrel));

  std::cout << "\n\nStudy Detector = Ecal SubDetector = EB"
            << "\n======================================\n\n";
  const std::vector<DetId>& ids = ecalGeom->getValidDetIds(DetId::Ecal, EcalBarrel);
  int nall(0);
  for (auto id : ids) {
    ++nall;
    std::shared_ptr<const CaloCellGeometry> geom = ecalGeom->getGeometry(id);
    EBDetId ebid(id.rawId());

    std::cout << "IEta = " << ebid.ieta() << ";  IPhi = " << ebid.iphi() << " geom->getPosition "
              << std::setprecision(4) << geom->getPosition() << " BackPoint " << geom->getBackPoint()
              << " [rho,eta:etaSpan,phi:phiSpan] (" << geom->rhoPos() << ", " << geom->etaPos() << ":"
              << geom->etaSpan() << ", " << geom->phiPos() << ":" << geom->phiSpan() << ")";

    const CaloCellGeometry::CornersVec& corners(geom->getCorners());

    for (unsigned int ci(0); ci != corners.size(); ci++) {
      std::cout << " Corner: " << ci << "  Location" << corners[ci] << " ; ";
    }

    std::cout << std::endl;
  }
  std::cout << "\n\nDumps a total of : " << nall << " cells of the detector\n" << std::endl;
}

DEFINE_FWK_MODULE(EcalBarrelCellParameterDump);
