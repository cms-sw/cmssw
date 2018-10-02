#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
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
  explicit HcalDetIdTester( const edm::ParameterSet& );
  ~HcalDetIdTester( void ) override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  bool              geomDB_;
};

HcalDetIdTester::HcalDetIdTester( const edm::ParameterSet& iConfig ) {
  geomDB_ = iConfig.getParameter<bool>("GeometryFromDB");
}

HcalDetIdTester::~HcalDetIdTester( void ) {}

void
HcalDetIdTester::analyze(const edm::Event& /*iEvent*/,
			 const edm::EventSetup& iSetup ) {

  edm::ESHandle<HcalDDDRecConstants> hDRCons;
  iSetup.get<HcalRecNumberingRecord>().get(hDRCons);
  const HcalDDDRecConstants hcons = (*hDRCons);
  edm::ESHandle<HcalTopology> topologyHandle;
  iSetup.get<HcalRecNumberingRecord>().get( topologyHandle );
  const HcalTopology topology = (*topologyHandle);

  CaloSubdetectorGeometry* caloGeom(nullptr);
  if (geomDB_) {
    edm::ESHandle<CaloGeometry> pG;
    iSetup.get<CaloGeometryRecord>().get(pG);
    const CaloGeometry* geo = pG.product();
    caloGeom = (CaloSubdetectorGeometry*)(geo->getSubdetectorGeometry(DetId::Hcal,HcalBarrel));
  } else {
    HcalFlexiHardcodeGeometryLoader m_loader;
    caloGeom = m_loader.load(topology, hcons);
  }

  int maxDepth = (topology.maxDepthHB() > topology.maxDepthHE()) ?
    topology.maxDepthHB() : topology.maxDepthHE();

  int nfail0(0);
  for (int det = 1; det <= HcalForward; det++) {
    for (int eta = -HcalDetId::kHcalEtaMask2;
	 eta <= HcalDetId::kHcalEtaMask2; eta++) {
      for (int depth = 1; depth <= maxDepth; depth++) {
	for (int phi = 0; phi <= HcalDetId::kHcalPhiMask2; phi++) {
	  HcalDetId detId ((HcalSubdetector) det, eta, phi, depth);
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
  for (auto id : ids)  {
    if (!topology.valid(id)) {
      std::cout << HcalDetId(id) << " declared as invalid == ERROR\n";
      ++nfail1;
    }
  }

  std::cout << "\nNumber of failures:\nTopology certifies but geometry fails "
	    << nfail0 << "\nGeometry certifies but Topology fails " << nfail1
	    << std::endl << std::endl;
}

DEFINE_FWK_MODULE(HcalDetIdTester);
