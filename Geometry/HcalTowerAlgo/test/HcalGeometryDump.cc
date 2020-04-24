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

class HcalGeometryDump : public edm::one::EDAnalyzer<> {

public:
  explicit HcalGeometryDump( const edm::ParameterSet& );
  ~HcalGeometryDump( void ) override;
    
  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  edm::ParameterSet ps0_;
  bool              geomDB_;
};

HcalGeometryDump::HcalGeometryDump( const edm::ParameterSet& iConfig ) : ps0_(iConfig) {
  geomDB_ = iConfig.getParameter<bool>("GeometryFromDB");
}

HcalGeometryDump::~HcalGeometryDump( void ) {}

void
HcalGeometryDump::analyze(const edm::Event& /*iEvent*/, 
			  const edm::EventSetup& iSetup ) {

  edm::ESHandle<HcalDDDRecConstants> hDRCons;
  iSetup.get<HcalRecNumberingRecord>().get(hDRCons);
  const HcalDDDRecConstants hcons = (*hDRCons);
  edm::ESHandle<HcalTopology> topologyHandle;
  iSetup.get<HcalRecNumberingRecord>().get( topologyHandle );
  const HcalTopology topology = (*topologyHandle);

  HcalGeometry* caloGeom(nullptr);
  if (geomDB_) {
    edm::ESHandle<CaloGeometry> pG;
    iSetup.get<CaloGeometryRecord>().get(pG);
    const CaloGeometry* geo = pG.product();
    caloGeom = (HcalGeometry*)(geo->getSubdetectorGeometry(DetId::Hcal,HcalBarrel));
  } else {
    HcalFlexiHardcodeGeometryLoader m_loader(ps0_);
    caloGeom = (HcalGeometry*)(m_loader.load(topology, hcons));
  }

  const std::vector<DetId>& ids = caloGeom->getValidDetIds();

  for (int subdet=1; subdet<= 4; ++subdet) {
    std::vector<unsigned int> detIds;
    for (auto id : ids)  {
      DetId hid = id;
      if (hid.subdetId() == subdet) {
	detIds.emplace_back(hid.rawId());
      }
    }
    std::cout << detIds.size() << " valid Ids for subdetector " << subdet
	      << std::endl;
    std::sort(detIds.begin(), detIds.end());
    int counter = 0;
    for (std::vector<unsigned int>::const_iterator i = detIds.begin();  
	 i != detIds.end(); ++i, ++counter)  {
      HcalDetId hid = HcalDetId(*i);
      const CaloCellGeometry * cell = caloGeom->getGeometry(*i);
      std::cout << hid << "\tCaloCellGeometry " << cell->getPosition() 
		<< "\tHcalGeometry " << caloGeom->getPosition(hid) 
		<< std::endl;
    }
  }
}

DEFINE_FWK_MODULE(HcalGeometryDump);
