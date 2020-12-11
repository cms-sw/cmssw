#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalFlexiHardcodeGeometryLoader.h"
#include "Geometry/HcalTowerAlgo/interface/HcalHardcodeGeometryLoader.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include <iostream>

class HcalGeometryAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit HcalGeometryAnalyzer(const edm::ParameterSet&);
  ~HcalGeometryAnalyzer(void) override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  bool useOld_;
  bool geomDB_;
  edm::ESGetToken<HcalDDDRecConstants, HcalRecNumberingRecord> tok_ddrec_;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> tok_htopo_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
};

HcalGeometryAnalyzer::HcalGeometryAnalyzer(const edm::ParameterSet& iConfig) {
  useOld_ = iConfig.getParameter<bool>("UseOldLoader");
  geomDB_ = iConfig.getParameter<bool>("GeometryFromDB");
  tok_ddrec_ = esConsumes<HcalDDDRecConstants, HcalRecNumberingRecord>();
  tok_htopo_ = esConsumes<HcalTopology, HcalRecNumberingRecord>();
  tok_geom_ = esConsumes<CaloGeometry, CaloGeometryRecord>();
}

HcalGeometryAnalyzer::~HcalGeometryAnalyzer(void) {}

void HcalGeometryAnalyzer::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {
  const HcalDDDRecConstants hcons = iSetup.getData(tok_ddrec_);
  const HcalTopology topology = iSetup.getData(tok_htopo_);

  CaloSubdetectorGeometry* caloGeom(nullptr);
  if (geomDB_) {
    const CaloGeometry* geo = &iSetup.getData(tok_geom_);
    caloGeom = (CaloSubdetectorGeometry*)(geo->getSubdetectorGeometry(DetId::Hcal, HcalBarrel));
  } else if (useOld_) {
    HcalHardcodeGeometryLoader m_loader;
    caloGeom = m_loader.load(topology);
  } else {
    HcalFlexiHardcodeGeometryLoader m_loader;
    caloGeom = m_loader.load(topology, hcons);
  }
  const std::vector<DetId>& ids = caloGeom->getValidDetIds();

  std::vector<int> dins;
  int counter = 0;
  for (std::vector<DetId>::const_iterator i = ids.begin(), iEnd = ids.end(); i != iEnd; ++i, ++counter) {
    HcalDetId hid = (*i);
    std::cout << counter << ": din " << topology.detId2denseId(*i) << ":" << hid;
    dins.emplace_back(topology.detId2denseId(*i));

    auto cell = caloGeom->getGeometry(*i);
    std::cout << *cell << std::endl;
  }

  std::sort(dins.begin(), dins.end());
  std::cout << "=== Total " << counter << " cells in HCAL."
            << " from HcalTopology ncells " << topology.ncells() << std::endl;

  // HB : 6911: din 16123
  std::cout << "HB Size " << topology.getHBSize() << "\nHE Size " << topology.getHESize() << "\nHO Size "
            << topology.getHOSize() << "\nHF Size " << topology.getHFSize() << "\nTotal "
            << topology.getHBSize() + topology.getHESize() + topology.getHOSize() + topology.getHFSize() << "\n";

  counter = 0;
  for (std::vector<int>::const_iterator i = dins.begin(); i != dins.end(); ++i, ++counter) {
    HcalDetId hid = (topology.denseId2detId(*i));
    HcalDetId ihid = (topology.denseId2detId(dins[counter]));
    std::cout << counter << ": din " << (*i) << " :" << hid << " == " << ihid << std::endl;
  }
}

DEFINE_FWK_MODULE(HcalGeometryAnalyzer);
