#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
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
  explicit HcalGeometryAnalyzer( const edm::ParameterSet& );
  ~HcalGeometryAnalyzer( void ) override;
    
  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  edm::ParameterSet ps0_;
  bool              useOld_;
  bool              geomDB_;
};

HcalGeometryAnalyzer::HcalGeometryAnalyzer( const edm::ParameterSet& iConfig ) : ps0_(iConfig) {
  useOld_ = iConfig.getParameter<bool>("UseOldLoader");
  geomDB_ = iConfig.getParameter<bool>("GeometryFromDB");
}

HcalGeometryAnalyzer::~HcalGeometryAnalyzer( void ) {}

void
HcalGeometryAnalyzer::analyze( const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup ) {

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
  } else if (useOld_) {
    HcalHardcodeGeometryLoader m_loader(ps0_);
    caloGeom = m_loader.load(topology);
  } else {
    HcalFlexiHardcodeGeometryLoader m_loader(ps0_);
    caloGeom = m_loader.load(topology, hcons);
  }
  const std::vector<DetId>& ids = caloGeom->getValidDetIds();

  std::vector<int> dins;
  int counter = 0;
  for( std::vector<DetId>::const_iterator i = ids.begin(), iEnd = ids.end(); i != iEnd; ++i, ++counter )  {
    HcalDetId hid = (*i);
    std::cout << counter << ": din " << topology.detId2denseId(*i) << ":" << hid;
    dins.emplace_back( topology.detId2denseId(*i));
	
    const CaloCellGeometry * cell = caloGeom->getGeometry(*i);
    std::cout << *cell << std::endl;
  }

  std::sort( dins.begin(), dins.end());
  std::cout << "=== Total " << counter << " cells in HCAL."
	    << " from HcalTopology ncells " << topology.ncells() << std::endl;

  // HB : 6911: din 16123
  std::cout << "HB Size " << topology.getHBSize()
	    << "\nHE Size " << topology.getHESize()
	    << "\nHO Size " << topology.getHOSize()
	    << "\nHF Size " << topology.getHFSize()
	    << "\nTotal " << topology.getHBSize() + topology.getHESize() + topology.getHOSize() + topology.getHFSize() 
	    << "\n";
    
  counter = 0;
  for (std::vector<int>::const_iterator i=dins.begin(); i != dins.end(); ++i, ++counter) {
    HcalDetId hid = (topology.denseId2detId(*i));
    HcalDetId ihid = (topology.denseId2detId(dins[counter]));
    std::cout << counter << ": din " << (*i) << " :" << hid << " == " << ihid << std::endl;
  }
}

DEFINE_FWK_MODULE(HcalGeometryAnalyzer);
