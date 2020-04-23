#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"

#include <Geometry/EcalMapping/interface/EcalElectronicsMapping.h>
#include <Geometry/EcalMapping/interface/EcalMappingRcd.h>

#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/EZArrayFL.h"

#include <iomanip>
#include <iostream>

typedef EZArrayFL<GlobalPoint> CornersVec;

class ecalBarrelCellParameterDump : public edm::one::EDAnalyzer<> {
public:
  explicit ecalBarrelCellParameterDump(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  static const int detMax_ = 4;
  int subdet_;
};

ecalBarrelCellParameterDump::ecalBarrelCellParameterDump(const edm::ParameterSet& iConfig) {
  subdet_ = std::min(detMax_, std::max(iConfig.getParameter<int>("SubDetector"), 1));
  subdet_ = std::min(detMax_, std::max(subdet_, 1));
}

void ecalBarrelCellParameterDump::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<int>("SubDetector", 1);
  descriptions.add("ecalBarrelCellParameterDump", desc);
}

void ecalBarrelCellParameterDump::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {
  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  const CaloGeometry* geo = pG.product();
  const CaloSubdetectorGeometry* ecalGeom =
      static_cast<const CaloSubdetectorGeometry*>(geo->getSubdetectorGeometry(DetId::Ecal, EcalBarrel));

  std::string subdets[detMax_] = {"EB"};

  std::cout << "\n\nStudy Detector = Ecal SubDetector = " << subdets[subdet_ - 1]
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

DEFINE_FWK_MODULE(ecalBarrelCellParameterDump);
