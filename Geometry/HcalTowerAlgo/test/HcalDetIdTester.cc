#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalFlexiHardcodeGeometryLoader.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include <iostream>

class HcalDetIdTester : public edm::one::EDAnalyzer<> {
public:
  explicit HcalDetIdTester(const edm::ParameterSet&);
  ~HcalDetIdTester(void) override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  bool geomDB_;
  edm::ESGetToken<HcalDDDRecConstants, HcalRecNumberingRecord> tok_ddrec_;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> tok_htopo_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
};

HcalDetIdTester::HcalDetIdTester(const edm::ParameterSet& iConfig) {
  geomDB_ = iConfig.getParameter<bool>("GeometryFromDB");
  tok_ddrec_ = esConsumes<HcalDDDRecConstants, HcalRecNumberingRecord>();
  tok_htopo_ = esConsumes<HcalTopology, HcalRecNumberingRecord>();
  tok_geom_ = esConsumes<CaloGeometry, CaloGeometryRecord>();
}

HcalDetIdTester::~HcalDetIdTester(void) {}

void HcalDetIdTester::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {
  const HcalDDDRecConstants hcons = iSetup.getData(tok_ddrec_);
  const HcalTopology topology = iSetup.getData(tok_htopo_);

  CaloSubdetectorGeometry* caloGeom(nullptr);
  if (geomDB_) {
    const CaloGeometry* geo = &iSetup.getData(tok_geom_);
    caloGeom = (CaloSubdetectorGeometry*)(geo->getSubdetectorGeometry(DetId::Hcal, HcalBarrel));
  } else {
    HcalFlexiHardcodeGeometryLoader m_loader;
    caloGeom = m_loader.load(topology, hcons);
  }

  int maxDepth = (topology.maxDepthHB() > topology.maxDepthHE()) ? topology.maxDepthHB() : topology.maxDepthHE();

  int nfail0(0);
  for (int det = 1; det <= HcalForward; det++) {
    for (int eta = -HcalDetId::kHcalEtaMask2; eta <= (int)(HcalDetId::kHcalEtaMask2); eta++) {
      for (int depth = 1; depth <= maxDepth; depth++) {
        for (unsigned int phi = 0; phi <= HcalDetId::kHcalPhiMask2; phi++) {
          HcalDetId detId((HcalSubdetector)det, eta, phi, depth);
          if (topology.valid(detId)) {
            auto cell = caloGeom->getGeometry((DetId)(detId));
            if (cell) {
              std::cout << detId << " " << cell->getPosition() << std::endl;
            } else {
              std::cout << detId << " position not found" << std::endl;
              ++nfail0;
            }
          }
        }
      }
    }
  }

  int nfail1(0);
  const std::vector<DetId>& ids = caloGeom->getValidDetIds();
  for (auto id : ids) {
    if (!topology.valid(id)) {
      std::cout << HcalDetId(id) << " declared as invalid == ERROR\n";
      ++nfail1;
    }
  }

  std::cout << "\nNumber of failures:\nTopology certifies but geometry fails " << nfail0
            << "\nGeometry certifies but Topology fails " << nfail1 << std::endl
            << std::endl;
}

DEFINE_FWK_MODULE(HcalDetIdTester);
