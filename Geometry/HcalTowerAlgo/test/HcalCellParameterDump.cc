#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include <iomanip>
#include <iostream>

class HcalCellParameterDump : public edm::one::EDAnalyzer<> {
public:
  explicit HcalCellParameterDump(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  static constexpr int detMax_ = 4;
  int subdet_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
};

HcalCellParameterDump::HcalCellParameterDump(const edm::ParameterSet& iConfig) {
  subdet_ = std::min(detMax_, std::max(iConfig.getParameter<int>("SubDetector"), 1));
  subdet_ = std::min(detMax_, std::max(subdet_, 1));
  tok_geom_ = esConsumes<CaloGeometry, CaloGeometryRecord>();
  std::cout << "Dumps all cells for SubDetId: " << subdet_ << std::endl;
}

void HcalCellParameterDump::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<int>("SubDetector", 1);
  descriptions.add("hcalCellParameterDump", desc);
}

void HcalCellParameterDump::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {
  const CaloGeometry* geo = &iSetup.getData(tok_geom_);
  const HcalGeometry* hcalGeom = static_cast<const HcalGeometry*>(geo->getSubdetectorGeometry(DetId::Hcal, HcalBarrel));

  std::string subdets[detMax_] = {"HB", "HE", "HO", "HF"};
  HcalSubdetector subdetd[detMax_] = {HcalBarrel, HcalEndcap, HcalOuter, HcalForward};

  std::cout << "\n\nStudy Detector = Hcal SubDetector = " << subdets[subdet_ - 1]
            << "\n======================================\n\n";
  const std::vector<DetId>& ids = hcalGeom->getValidDetIds(DetId::Hcal, subdetd[subdet_ - 1]);
  int nall(0);
  for (auto id : ids) {
    ++nall;
    std::shared_ptr<const CaloCellGeometry> geom = hcalGeom->getGeometry(id);
    std::cout << "[" << nall << "] " << HcalDetId(id) << " Reference " << std::setprecision(4) << geom->getPosition()
              << " Back " << geom->getBackPoint() << " [r,eta,phi] (" << geom->rhoPos() << ", " << geom->etaPos() << ":"
              << geom->etaSpan() << ", " << geom->phiPos() << ":" << geom->phiSpan() << ")\n";
  }
  std::cout << "\n\nDumps" << nall << " cells of the detector\n" << std::endl;
}

DEFINE_FWK_MODULE(HcalCellParameterDump);
