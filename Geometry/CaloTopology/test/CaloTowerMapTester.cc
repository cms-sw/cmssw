#include <iostream>
#include <string>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTowerConstituentsMap.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

class CaloTowerMapTester : public edm::EDAnalyzer {
public:
  explicit CaloTowerMapTester(const edm::ParameterSet& );
  
  void analyze(const edm::Event&, const edm::EventSetup& ) override;
  void doTest(const CaloGeometry* geo, const CaloTowerConstituentsMap* ctmap);

private:
  // ----------member data ---------------------------
};

CaloTowerMapTester::CaloTowerMapTester(const edm::ParameterSet& ) {}

void CaloTowerMapTester::analyze(const edm::Event& , 
				 const edm::EventSetup& iSetup ) {
  edm::ESHandle<CaloGeometry>             pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
  edm::ESHandle<CaloTowerConstituentsMap> ct;
  iSetup.get<CaloGeometryRecord>().get(ct);
  if (pG.isValid() && ct.isValid()) doTest(pG.product(),ct.product());
  else std::cout << "CaloGeometry in EventSetup " << pG.isValid() 
		 << " and CaloTowerConstituentsMap " << ct.isValid()
		 << std::endl;
}

void CaloTowerMapTester::doTest(const CaloGeometry* geo,
				const CaloTowerConstituentsMap* ctmap) {

  HcalGeometry* hgeo = (HcalGeometry*)(geo->getSubdetectorGeometry(DetId::Hcal,HcalBarrel));
  const std::vector<DetId>& dets = hgeo->getValidDetIds(DetId::Hcal,0);

  for (const auto& id: dets) {
    CaloTowerDetId tower = ctmap->towerOf(id);
    std::vector<DetId> ids = ctmap->constituentsOf(tower);
    std::cout << HcalDetId(id) << " belongs to " << tower << " which has "
	      << ids.size() << " constituents\n";
    for (unsigned int i=0; i<ids.size(); ++i) {
      std::cout << "[" << i << "] " << std::hex << ids[i].rawId() << std::dec;
      if        (ids[i].det()==DetId::Ecal && ids[i].subdetId()==EcalBarrel) {
	std::cout << " " << EBDetId(ids[i]) << std::endl;
      } else if (ids[i].det()==DetId::Ecal && ids[i].subdetId()==EcalEndcap) {
	std::cout << " " << EEDetId(ids[i]) << std::endl;
      } else if (ids[i].det()==DetId::Ecal && ids[i].subdetId()==EcalPreshower) {
	std::cout << " " << ESDetId(ids[i]) << std::endl;
      } else if (ids[i].det()==DetId::Hcal) {
	std::cout << " " << HcalDetId(ids[i]) << std::endl;
      } else {
	std::cout << std::endl;
      }
    }
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(CaloTowerMapTester);
