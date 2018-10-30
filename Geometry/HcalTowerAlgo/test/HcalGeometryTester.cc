#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalFlexiHardcodeGeometryLoader.h"
#include "Geometry/HcalTowerAlgo/interface/HcalHardcodeGeometryLoader.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include <iostream>
#include <string>

class HcalGeometryTester : public edm::one::EDAnalyzer<> {

public:
  explicit HcalGeometryTester( const edm::ParameterSet& );
  ~HcalGeometryTester( void ) override {}
    
  void beginJob() override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override {}

private:
  void testValidDetIds(CaloSubdetectorGeometry* geom, const HcalTopology& topo, 
		       DetId::Detector det, int subdet, const std::string& label);
  void testClosestCells(CaloSubdetectorGeometry* geom, const HcalTopology& top);
  void testClosestCell(const HcalDetId & detId, CaloSubdetectorGeometry * geom);
  void testTriggerGeometry(const HcalTopology& topology);
  void testFlexiValidDetIds(CaloSubdetectorGeometry* geom, 
			    const HcalTopology& topology, DetId::Detector det, 
			    int subdet, const std::string& label, 
			    std::vector<int> &dins);
  void testFlexiGeomHF(CaloSubdetectorGeometry* geom);

  bool              useOld_;
};

HcalGeometryTester::HcalGeometryTester( const edm::ParameterSet& iConfig ) {
  useOld_ = iConfig.getParameter<bool>( "UseOldLoader" );
}

void HcalGeometryTester::analyze(const edm::Event& /*iEvent*/, 
				 const edm::EventSetup& iSetup) {

  edm::ESHandle<HcalDDDRecConstants> hDRCons;
  iSetup.get<HcalRecNumberingRecord>().get(hDRCons);
  const HcalDDDRecConstants hcons = (*hDRCons);
  edm::ESHandle<HcalTopology> topologyHandle;
  iSetup.get<HcalRecNumberingRecord>().get(topologyHandle);
  const HcalTopology topology = (*topologyHandle);
  CaloSubdetectorGeometry* geom(nullptr);
  if (useOld_) {
    HcalHardcodeGeometryLoader m_loader;
    geom = m_loader.load(topology);
  } else {
    HcalFlexiHardcodeGeometryLoader m_loader;
    geom = m_loader.load(topology, hcons);
  }

  testValidDetIds(geom, topology, DetId::Hcal, HcalBarrel,  " BARREL ");
  testValidDetIds(geom, topology, DetId::Hcal, HcalEndcap,  " ENDCAP ");
  testValidDetIds(geom, topology, DetId::Hcal, HcalOuter,   " OUTER ");
  testValidDetIds(geom, topology, DetId::Hcal, HcalForward, " FORWARD ");

  testTriggerGeometry(topology);

  testClosestCells(geom, topology);

  std::cout << "HcalGeometryTester::Test SLHC Hcal Geometry" << std::endl;
  std::vector<int> dins;

  testFlexiValidDetIds(geom, topology, DetId::Hcal, HcalBarrel, " BARREL ",  dins );
  testFlexiValidDetIds(geom, topology, DetId::Hcal, HcalEndcap, " ENDCAP ",  dins );
  testFlexiValidDetIds(geom, topology, DetId::Hcal, HcalOuter,  " OUTER ",   dins );
  testFlexiValidDetIds(geom, topology, DetId::Hcal, HcalForward," FORWARD ", dins );

  testFlexiGeomHF(geom);

}

void HcalGeometryTester::testValidDetIds(CaloSubdetectorGeometry* caloGeom,
					 const HcalTopology& topology, 
					 DetId::Detector det, int subdet, 
					 const std::string& label) {

  std::stringstream s;
  s << label << " : " << std::endl;
  const std::vector<DetId>& idshb = caloGeom->getValidDetIds(det, subdet);
 
  int counter = 0;
  for (std::vector<DetId>::const_iterator i=idshb.begin(); i!=idshb.end(); 
       i++, ++counter) {
    HcalDetId hid=(*i);
    s << counter << ": din " << topology.detId2denseId(*i) << ":" 
	      << hid;
    auto cell = caloGeom->getGeometry(*i);
    s << *cell << std::endl;
  }
 
  s << "=== Total " << counter << " cells in " << label << std::endl;
  std::cout << s.str();
}

void HcalGeometryTester::testClosestCells(CaloSubdetectorGeometry* g,
					  const HcalTopology& topology ) {
  
  // make sure each cell is its own closest cell
  HcalDetId barrelDet1(HcalBarrel, 1, 1, 1);
  HcalDetId barrelDet2(HcalBarrel, 16, 50, 1);
  HcalDetId endcapDet1(HcalEndcap, -17, 72, 1);
  HcalDetId endcapDet2(HcalEndcap, 29, 35, 1);
  HcalDetId forwardDet1(HcalForward, 30, 71, 1);
  HcalDetId forwardDet3(HcalForward, -40, 71, 1);
  
  if (topology.valid(barrelDet1))  testClosestCell(barrelDet1 , g);
  if (topology.valid(barrelDet2))  testClosestCell(barrelDet2 , g);
  if (topology.valid(endcapDet1))  testClosestCell(endcapDet1 , g);
  if (topology.valid(endcapDet2))  testClosestCell(endcapDet2 , g);
  if (topology.valid(forwardDet1)) testClosestCell(forwardDet1, g);
  if (topology.valid(forwardDet3)) testClosestCell(forwardDet3, g);
  
  const std::vector<DetId>& idsb=g->getValidDetIds(DetId::Hcal,HcalBarrel);
  for (auto id : idsb) {
    testClosestCell(HcalDetId(id), g);
  }
  
  const std::vector<DetId>& idse=g->getValidDetIds(DetId::Hcal,HcalEndcap);
  for (auto id : idse) {
    testClosestCell(HcalDetId(id), g);
  }
}

void HcalGeometryTester::testClosestCell(const HcalDetId & detId, 
					 CaloSubdetectorGeometry *geom) {

  auto cell = geom->getGeometry(detId);
  HcalDetId closest = geom->getClosestCell(cell->getPosition());
  std::cout << "i/p " << detId << " position " << cell->getPosition() 
	    << " closest " << closest << std::endl;

  if(closest != detId) {
    std::cout << "HcalGeometryTester::Mismatch.  Original HCAL cell is "
	      << detId << " while nearest is " << closest << " ***ERROR***"
	      << std::endl;
  }
}

void HcalGeometryTester::testTriggerGeometry(const HcalTopology& topology) {

  HcalTrigTowerGeometry trigTowers( &topology );
  std::cout << "HCAL trigger tower eta bounds:" << std::endl;
  for(int ieta = 1; ieta <= 32; ++ieta) {
    double eta1, eta2;
    trigTowers.towerEtaBounds(ieta, 0, eta1, eta2);
    std::cout << "[" << ieta << "] "  << eta1 << " " << eta2 << std::endl;
  }

  // now test some cell mappings
  HcalDetId barrelDet(HcalBarrel, 1, 1, 1);
  HcalDetId endcapDet(HcalEndcap, 29, 1, 1);
  HcalDetId forwardDet1(HcalForward, 29, 71, 1);
  HcalDetId forwardDet2(HcalForward, 29, 71, 2);
  HcalDetId forwardDet3(HcalForward, 40, 71, 1);

  using TowerDets = std::vector<HcalTrigTowerDetId>;
  if (topology.valid(barrelDet)) {
    TowerDets barrelTowers = trigTowers.towerIds(barrelDet);
    std::cout << "Trigger Tower Size: Barrel " << barrelTowers.size()
	       << std::endl;
    assert(barrelTowers.size() ==1);
    std::cout << "Tower[0] " << barrelTowers[0] << std::endl;
  } 
  if (topology.valid(endcapDet)) {
    TowerDets endcapTowers = trigTowers.towerIds(endcapDet);
    std::cout << "Trigger Tower Size: Endcap " << endcapTowers.size() 
	      << std::endl;
    assert(!endcapTowers.empty());
    for (unsigned int k=0; k<endcapTowers.size(); ++k)
      std::cout << "Tower[" << k << "] " << endcapTowers[k] << std::endl;
  }
  if (topology.valid(forwardDet1)) {
    TowerDets forwardTowers1 = trigTowers.towerIds(forwardDet1);
    std::cout << "Trigger Tower Size: Forward1 " << forwardTowers1.size()
	      << std::endl;
    assert(!forwardTowers1.empty());
    for (unsigned int k=0; k<forwardTowers1.size(); ++k)
      std::cout << "Tower[" << k << "] " << forwardTowers1[k] << std::endl;
  }
  if (topology.valid(forwardDet2)) {
    TowerDets forwardTowers2 = trigTowers.towerIds(forwardDet2);
    std::cout << "Trigger Tower Size: Forward2 " << forwardTowers2.size()
	      << std::endl;
    assert(!forwardTowers2.empty());
    for (unsigned int k=0; k<forwardTowers2.size(); ++k)
      std::cout << "Tower[" << k << "] " << forwardTowers2[k] << std::endl;
  }
  if (topology.valid(forwardDet3)) {
    TowerDets forwardTowers3 = trigTowers.towerIds(forwardDet3);
    std::cout << "Trigger Tower Size: Forward3 " << forwardTowers3.size()
	      << std::endl;
    assert(!forwardTowers3.empty());
    for (unsigned int k=0; k<forwardTowers3.size(); ++k)
      std::cout << "Tower[" << k << "] " << forwardTowers3[k] << std::endl;
  }
}

void HcalGeometryTester::testFlexiValidDetIds(CaloSubdetectorGeometry* caloGeom,
					      const HcalTopology& topology, 
					      DetId::Detector det, int subdet, 
					      const std::string& label, 
					      std::vector<int> &dins) {

  std::stringstream s;
  s << label << " : " << std::endl;
  const std::vector<DetId>& idshb = caloGeom->getValidDetIds(det, subdet);
    
  int counter = 0;
  for (std::vector<DetId>::const_iterator i=idshb.begin(); i!=idshb.end(); 
       i++, ++counter) {
    HcalDetId hid=(*i);
    s << counter << ": din " << topology.detId2denseId(*i) << ":" << hid;
    dins.emplace_back( topology.detId2denseId(*i));
	
    auto cell = caloGeom->getGeometry(*i);
    s << *cell << std::endl;
  }

  std::sort( dins.begin(), dins.end());
  s << "=== Total " << counter << " cells in " << label << std::endl;

  counter = 0;
  for (std::vector<int>::const_iterator i=dins.begin(); i != dins.end(); 
       ++i, ++counter) {
    HcalDetId hid = (topology.denseId2detId(*i));
    HcalDetId ihid = (topology.denseId2detId(dins[counter]));
    s << counter << ": din " << (*i) << " :" << hid << " == " << ihid << std::endl;
  }
  std::cout << s.str();
}

void HcalGeometryTester::testFlexiGeomHF(CaloSubdetectorGeometry* caloGeom) {

  std::stringstream s;
  s << "Test HF Geometry : " << std::endl;
  for (int ieta = 29; ieta <=41; ++ieta) {
    HcalDetId cell3 (HcalForward, ieta, 3, 1);
    auto cellGeometry3 = caloGeom->getGeometry (cell3);
    if (cellGeometry3) {
      s << "cell geometry iphi=3 -> ieta=" << ieta
	<< " eta " << cellGeometry3->getPosition().eta () << "+-" 
	<< std::abs(cellGeometry3->getCorners()[0].eta() -
		    cellGeometry3->getCorners()[2].eta())/2
	<< " phi " << cellGeometry3->getPosition().phi ()/3.1415*180 
	<<  "+-" << std::abs(cellGeometry3->getCorners()[0].phi() -
			     cellGeometry3->getCorners()[2].phi())/3.1415*180/2.
	<< std::endl;
    }
  }
  std::cout << s.str();
}

DEFINE_FWK_MODULE(HcalGeometryTester);
