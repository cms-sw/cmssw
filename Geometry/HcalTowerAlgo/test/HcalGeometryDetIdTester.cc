#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include <iostream>

class HcalGeometryDetIdTester : public edm::one::EDAnalyzer<> {
public:
  explicit HcalGeometryDetIdTester(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  static constexpr int ndetMax_ = 4;
  int detMin_, detMax_;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> tok_htopo_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
};

HcalGeometryDetIdTester::HcalGeometryDetIdTester(const edm::ParameterSet& iConfig) {
  detMin_ = std::min(ndetMax_, std::max(iConfig.getParameter<int>("DetectorMin"), 1));
  detMax_ = std::min(ndetMax_, std::max(iConfig.getParameter<int>("DetectorMax"), 1));
  if (detMin_ > detMax_)
    detMin_ = detMax_;
  tok_htopo_ = esConsumes<HcalTopology, HcalRecNumberingRecord>();
  tok_geom_ = esConsumes<CaloGeometry, CaloGeometryRecord>();
  std::cout << "Study DetIds for SubDetId in the range " << detMin_ << ":" << detMax_ << std::endl;
}

void HcalGeometryDetIdTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<int>("DetectorMin", 1);
  desc.add<int>("DetectorMax", 4);
  descriptions.add("hcalGeometryDetIdTester", desc);
}

void HcalGeometryDetIdTester::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {
  const HcalTopology topology = iSetup.getData(tok_htopo_);
  const CaloGeometry* geo = &iSetup.getData(tok_geom_);
  const HcalGeometry* hcalGeom = static_cast<const HcalGeometry*>(geo->getSubdetectorGeometry(DetId::Hcal, HcalBarrel));

  std::string subdets[ndetMax_] = {"HB", "HE", "HO", "HF"};
  HcalSubdetector subdetd[ndetMax_] = {HcalBarrel, HcalEndcap, HcalOuter, HcalForward};
  int ietaMin[ndetMax_] = {1, 16, 1, 29};
  int ietaMax[ndetMax_] = {16, 29, 15, 41};
  int depthMin[ndetMax_] = {1, 1, 4, 1};
  int depthMax[ndetMax_] = {4, 7, 4, 4};

  for (int subd = (detMin_ - 1); subd < detMax_; ++subd) {
    std::cout << "\n\nStudy Detector = Hcal SubDetector = " << subdets[subd]
              << "\n======================================\n\n";
    int nall(0), nbad(0);
    const std::vector<DetId>& ids = hcalGeom->getValidDetIds(DetId::Hcal, subdetd[subd]);
    for (auto id : ids) {
      ++nall;
      if (!(topology.valid(id))) {
        ++nbad;
        std::cout << "Check " << HcalDetId(id) << " *****\n";
      }
    }
    std::cout << "\n"
              << nbad << " bad out of " << nall
              << " detIds\n========================\n\nNow List All IDs\n================\n";
    int k(0);
    for (auto id : ids) {
      std::cout << "[ " << std::setw(4) << k << "] " << HcalDetId(id) << "\n";
      ++k;
    }

    int n(0);
    std::cout << "\nNow List all IDs declared valid by Topology\n===========================================\n\n";
    for (int ieta = ietaMin[subd]; ieta <= ietaMax[subd]; ++ieta) {
      for (int depth = depthMin[subd]; depth <= depthMax[subd]; ++depth) {
        HcalDetId id(subdetd[subd], ieta, 1, depth);
        if (topology.validHcal(id)) {
          std::cout << "[ " << std::setw(2) << n << "] " << id << "\n";
          ++n;
        }
      }
    }
    std::cout << "\nFinds a total of " << n << " IDs\n";
  }
}

DEFINE_FWK_MODULE(HcalGeometryDetIdTester);
