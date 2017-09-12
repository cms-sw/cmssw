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
#include <iostream>

class HcalGeometryDetIdAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit HcalGeometryDetIdAnalyzer( const edm::ParameterSet& );
  ~HcalGeometryDetIdAnalyzer( void ) override;
    
  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  edm::ParameterSet ps0_;
  bool              useOld_;
};

HcalGeometryDetIdAnalyzer::HcalGeometryDetIdAnalyzer(const edm::ParameterSet& iConfig ) : ps0_(iConfig) {
  useOld_ = iConfig.getParameter<bool>("UseOldLoader");
}

HcalGeometryDetIdAnalyzer::~HcalGeometryDetIdAnalyzer( void ) {}

void
HcalGeometryDetIdAnalyzer::analyze( const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup ) {
  edm::ESHandle<HcalDDDRecConstants> hDRCons;
  iSetup.get<HcalRecNumberingRecord>().get(hDRCons);
  const HcalDDDRecConstants hcons = (*hDRCons);
  edm::ESHandle<HcalTopology> topologyHandle;
  iSetup.get<HcalRecNumberingRecord>().get( topologyHandle );
  const HcalTopology topology = (*topologyHandle);

  CaloSubdetectorGeometry* caloGeom(nullptr);
  if (useOld_) {
    HcalHardcodeGeometryLoader m_loader(ps0_);
    caloGeom = m_loader.load(topology);
  } else {
    HcalFlexiHardcodeGeometryLoader m_loader(ps0_);
    caloGeom = m_loader.load(topology, hcons);
  }
  const std::vector<DetId>& ids = caloGeom->getValidDetIds();

  int counter = 0;
  for (std::vector<DetId>::const_iterator i = ids.begin(), iEnd = ids.end();
       i != iEnd; ++i, ++counter )  {
    HcalDetId hid = (*i);
    unsigned int did = topology.detId2denseId(*i);
    HcalDetId rhid = topology.denseId2detId(did);
	
    std::cout << counter << ": din " << std::hex << did << std::dec << ": " 
	      << hid << " == " << rhid << std::endl;
    assert(hid == rhid);
  }
}

DEFINE_FWK_MODULE(HcalGeometryDetIdAnalyzer);
