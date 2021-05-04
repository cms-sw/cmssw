#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/EZArrayFL.h"

#include "TH2.h"

#include <iomanip>
#include <iostream>
#include <sstream>  // for ostringstream

typedef EZArrayFL<GlobalPoint> CornersVec;

class EcalPreshowerCellParameterDump : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit EcalPreshowerCellParameterDump(const edm::ParameterSet&);
  ~EcalPreshowerCellParameterDump() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  const bool debug_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
  std::vector<TH2D*> hist_;
};

EcalPreshowerCellParameterDump::EcalPreshowerCellParameterDump(const edm::ParameterSet& ps)
    : debug_(ps.getUntrackedParameter<bool>("debug", false)),
      tok_geom_(esConsumes<CaloGeometry, CaloGeometryRecord>()) {
  usesResource(TFileService::kSharedResource);

  if (debug_) {
    edm::Service<TFileService> fs;
    for (short iz = 0; iz < 2; ++iz) {
      short zside = 2 * iz - 1;
      for (short lay = 1; lay <= 2; ++lay) {
        std::ostringstream name, title;
        name << "hist" << iz << lay;
        title << "y vs. x (zside = " << zside << ",layer = " << lay << ")";
        hist_.emplace_back(
            fs->make<TH2D>(name.str().c_str(), title.str().c_str(), 5000, -125.0, 125.0, 5000, -125.0, 125.0));
      }
    }
  }
}

void EcalPreshowerCellParameterDump::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<bool>("debug", false);
  descriptions.add("ecalPreshowerCellParameterDump", desc);
}

void EcalPreshowerCellParameterDump::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {
  const CaloGeometry* geo = &iSetup.getData(tok_geom_);
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

    if (debug_) {
      std::cout << nall << " " << esid.rawId() << " " << std::setprecision(6) << geom->getPosition() << std::endl;
      unsigned int hid = ((esid.zside() + 1) + esid.plane() - 1);
      if (hid < hist_.size())
        hist_[hid]->Fill(geom->getPosition().x(), geom->getPosition().y());
    } else {
      std::cout << "Cell[" << nall << "] " << esid << " geom->getPosition " << std::setprecision(4)
                << geom->getPosition() << " BackPoint " << geom->getBackPoint() << " [rho,eta:etaSpan,phi:phiSpan] ("
                << geom->rhoPos() << ", " << geom->etaPos() << ":" << geom->etaSpan() << ", " << geom->phiPos() << ":"
                << geom->phiSpan() << ")";

      const CaloCellGeometry::CornersVec& corners(geom->getCorners());

      for (unsigned int ci(0); ci != corners.size(); ci++) {
        std::cout << " Corner: " << ci << "  Location" << corners[ci] << " ; ";
      }

      std::cout << std::endl;
    }
  }
  std::cout << "\n\nDumps a total of : " << nall << " cells of the detector\n" << std::endl;
}

DEFINE_FWK_MODULE(EcalPreshowerCellParameterDump);
