#include "DataFormats/EcalDetId/interface/EEDetId.h"
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
#include <sstream>

typedef EZArrayFL<GlobalPoint> CornersVec;

class EcalEndcapCellParameterDump : public edm::one::EDAnalyzer<> {
public:
  explicit EcalEndcapCellParameterDump(const edm::ParameterSet&);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
};

EcalEndcapCellParameterDump::EcalEndcapCellParameterDump(const edm::ParameterSet&)
    : tok_geom_(esConsumes<CaloGeometry, CaloGeometryRecord>()) {}

void EcalEndcapCellParameterDump::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {
  const CaloGeometry* geo = &iSetup.getData(tok_geom_);
  const CaloSubdetectorGeometry* ecalGeom =
      static_cast<const CaloSubdetectorGeometry*>(geo->getSubdetectorGeometry(DetId::Ecal, EcalEndcap));

  edm::LogVerbatim("EcalGeom") << "\n\nStudy Detector = Ecal SubDetector = EE"
                               << "\n======================================\n";
  const std::vector<DetId>& ids = ecalGeom->getValidDetIds(DetId::Ecal, EcalEndcap);
  int nall(0);
  for (auto id : ids) {
    ++nall;
    std::shared_ptr<const CaloCellGeometry> geom = ecalGeom->getGeometry(id);
    EEDetId ebid(id.rawId());

    std::ostringstream st1;
    st1 << "IX = " << ebid.ix() << ";  IY = " << ebid.iy() << " geom->getPosition " << std::setprecision(4)
        << geom->getPosition() << " BackPoint " << geom->getBackPoint() << " [rho,eta:etaSpan,phi:phiSpan] ("
        << geom->rhoPos() << ", " << geom->etaPos() << ":" << geom->etaSpan() << ", " << geom->phiPos() << ":"
        << geom->phiSpan() << ")";

    const CaloCellGeometry::CornersVec& corners(geom->getCorners());

    for (unsigned int ci(0); ci != corners.size(); ci++) {
      st1 << " Corner: " << ci << "  Location" << corners[ci] << " ; ";
    }

    edm::LogVerbatim("EcalGeom") << st1.str();
  }
  edm::LogVerbatim("EcalGeom") << "\n\nDumps a total of : " << nall << " cells of the detector\n";
}

DEFINE_FWK_MODULE(EcalEndcapCellParameterDump);
