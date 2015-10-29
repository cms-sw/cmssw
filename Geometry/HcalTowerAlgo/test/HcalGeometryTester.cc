#include "FWCore/Framework/interface/EDAnalyzer.h"
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

class HcalGeometryTester : public edm::EDAnalyzer {

public:
  explicit HcalGeometryTester( const edm::ParameterSet& );
  ~HcalGeometryTester( void );
    
  virtual void analyze( const edm::Event&, const edm::EventSetup& );

private:
  void testValidDetIds(CaloSubdetectorGeometry* geom, const HcalTopology& topo, 
		       DetId::Detector det, int subdet, std::string label);
  void testClosestCells(CaloSubdetectorGeometry* geom, const HcalTopology& top);
  void testClosestCell(const HcalDetId & detId, CaloSubdetectorGeometry * geom);
  void testTriggerGeometry(const HcalTopology& topology);
  void testFlexiValidDetIds(CaloSubdetectorGeometry* geom, 
			    const HcalTopology& topology, DetId::Detector det, 
			    int subdet, std::string label, 
			    std::vector<int> &dins);
  void testFlexiGeomHF(CaloSubdetectorGeometry* geom);

  edm::ParameterSet ps0;
  std::string m_label;
  bool        useOld_;
};

HcalGeometryTester::HcalGeometryTester( const edm::ParameterSet& iConfig ) :
  ps0(iConfig),  m_label("_master") {
  m_label = iConfig.getParameter<std::string>( "HCALGeometryLabel" );
  useOld_ = iConfig.getParameter<bool>( "UseOldLoader" );
}

HcalGeometryTester::~HcalGeometryTester() { }

void HcalGeometryTester::analyze(const edm::Event& /*iEvent*/, 
				 const edm::EventSetup& iSetup) {

  edm::ESHandle<HcalDDDRecConstants> hDRCons;
  iSetup.get<HcalRecNumberingRecord>().get(hDRCons);
  const HcalDDDRecConstants hcons = (*hDRCons);
  edm::ESHandle<HcalTopology> topologyHandle;
  iSetup.get<HcalRecNumberingRecord>().get(topologyHandle);
  const HcalTopology topology = (*topologyHandle);
  CaloSubdetectorGeometry* geom(0);
  if (useOld_) {
    HcalHardcodeGeometryLoader m_loader(ps0);
    geom = m_loader.load(topology);
  } else {
    HcalFlexiHardcodeGeometryLoader m_loader(ps0);
    geom = m_loader.load(topology, hcons);
  }

  testValidDetIds(geom, topology, DetId::Hcal, HcalBarrel,  " BARREL ");
  testValidDetIds(geom, topology, DetId::Hcal, HcalEndcap,  " ENDCAP ");
  testValidDetIds(geom, topology, DetId::Hcal, HcalOuter,   " OUTER ");
  testValidDetIds(geom, topology, DetId::Hcal, HcalForward, " FORWARD ");

  testTriggerGeometry(topology);

  testClosestCells(geom, topology);
  std::cout << "Test SLHC Hcal Flexi geometry" << std::endl;
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
					 std::string label) {

  std::cout << std::endl << label << " : " << std::endl;
  const std::vector<DetId>& idshb = caloGeom->getValidDetIds(det, subdet);
 
  int counter = 0;
  for (std::vector<DetId>::const_iterator i=idshb.begin(); i!=idshb.end(); 
       i++, ++counter) {
    HcalDetId hid=(*i);
    std::cout << counter << ": din " << topology.detId2denseId(*i) << ":" 
	      << hid;
    const CaloCellGeometry * cell = caloGeom->getGeometry(*i);
    std::cout << *cell << std::endl;
  }
 
  std::cout << "=== Total " << counter << " cells in " << label << std::endl;
}

void HcalGeometryTester::testClosestCells(CaloSubdetectorGeometry* g,
					  const HcalTopology& topology ) {
  
  // make sure each cel is its own closest cell
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
  
  const std::vector<DetId>& ids=g->getValidDetIds(DetId::Hcal,HcalBarrel);
  for (std::vector<DetId>::const_iterator i=ids.begin(); i!=ids.end(); i++) {
    testClosestCell(HcalDetId(*i), g);
  }
}

void HcalGeometryTester::testClosestCell(const HcalDetId & detId, 
					 CaloSubdetectorGeometry *geom) {

  const CaloCellGeometry* cell = geom->getGeometry(detId);
  std::cout << "i/p " << detId << " position " << cell->getPosition();
  HcalDetId closest = geom->getClosestCell(cell->getPosition());
  std::cout << " closest " << closest << std::endl;

  if(closest != detId) {
    std::cout << "ERROR mismatch.  Original HCAL cell is "
	      << detId << " while nearest is " << closest << std::endl;
  }
}

void HcalGeometryTester::testTriggerGeometry(const HcalTopology& topology) {

  HcalTrigTowerGeometry trigTowers( &topology );
  std::cout << "HCAL trigger tower eta bounds " << std::endl;
  for(int ieta = 1; ieta <= 32; ++ieta) {
    double eta1, eta2;
    trigTowers.towerEtaBounds(ieta, eta1, eta2);
    std::cout << ieta << " "  << eta1 << " " << eta2 << std::endl;
  }

  // now test some cell mappings
  HcalDetId barrelDet(HcalBarrel, 1, 1, 1);
  HcalDetId endcapDet(HcalEndcap, 29, 1, 1);
  HcalDetId forwardDet1(HcalForward, 29, 71, 1);
  HcalDetId forwardDet2(HcalForward, 29, 71, 2);
  HcalDetId forwardDet3(HcalForward, 40, 71, 1);

  typedef std::vector<HcalTrigTowerDetId> TowerDets;
  if (topology.valid(barrelDet)) {
    TowerDets barrelTowers = trigTowers.towerIds(barrelDet);
    std::cout << "Trigger Tower Size: Barrel " << barrelTowers.size() << "\n";
    assert(barrelTowers.size() ==1);
    std::cout << barrelTowers[0] << std::endl;
  } 
  if (topology.valid(endcapDet)) {
    TowerDets endcapTowers = trigTowers.towerIds(endcapDet);
    std::cout << "Trigger Tower Size: Endcap " << endcapTowers.size() << "\n";
    assert(endcapTowers.size() >=1);
    std::cout << endcapTowers[0] << std::endl;
    if (endcapTowers.size() > 1) std::cout << endcapTowers[1] << std::endl;
  }
  if (topology.valid(forwardDet1)) {
    TowerDets forwardTowers1 = trigTowers.towerIds(forwardDet1);
    std::cout << "Trigger Tower Size: Forward1 " << forwardTowers1.size() << "\n";
    assert(forwardTowers1.size() ==1);
    std::cout << forwardTowers1[0] << std::endl;
  }
  if (topology.valid(forwardDet1)) {
    TowerDets forwardTowers2 = trigTowers.towerIds(forwardDet2);
    std::cout << "Trigger Tower Size: Forward2 " << forwardTowers2.size() << "\n";
    assert(forwardTowers2.size() ==1);
    std::cout << forwardTowers2[0] << std::endl;
  }
  if (topology.valid(forwardDet1)) {
    TowerDets forwardTowers3 = trigTowers.towerIds(forwardDet3);
    std::cout << "Trigger Tower Size: Forward3 " << forwardTowers3.size() << "\n";
    assert(forwardTowers3.size() ==1);
    std::cout << forwardTowers3[0] << std::endl;
  }
}

void HcalGeometryTester::testFlexiValidDetIds(CaloSubdetectorGeometry* caloGeom,
					      const HcalTopology& topology, 
					      DetId::Detector det, int subdet, 
					      std::string label, 
					      std::vector<int> &dins) {

  std::cout << std::endl << label << " : " << std::endl;
  const std::vector<DetId>& idshb = caloGeom->getValidDetIds(det, subdet);
    
  int counter = 0;
  for (std::vector<DetId>::const_iterator i=idshb.begin(); i!=idshb.end(); 
       i++, ++counter) {
    HcalDetId hid=(*i);
    std::cout << counter << ": din " << topology.detId2denseId(*i) << ":" << hid;
    dins.push_back( topology.detId2denseId(*i));
	
    const CaloCellGeometry * cell = caloGeom->getGeometry(*i);
    std::cout << *cell << std::endl;
  }

  std::sort( dins.begin(), dins.end());
  std::cout << "=== Total " << counter << " cells in " << label << std::endl;

  counter = 0;
  for (std::vector<int>::const_iterator i=dins.begin(); i != dins.end(); 
       ++i, ++counter) {
    HcalDetId hid = (topology.denseId2detId(*i));
    HcalDetId ihid = (topology.denseId2detId(dins[counter]));
    std::cout << counter << ": din " << (*i) << " :" << hid << " == " << ihid << std::endl;
  }
}

void HcalGeometryTester::testFlexiGeomHF(CaloSubdetectorGeometry* caloGeom) {

  std::cout << std::endl << "Test HF Geometry : " << std::endl;
  for (int ieta = 29; ieta <=41; ++ieta) {
    HcalDetId cell3 (HcalForward, ieta, 3, 1);
    const CaloCellGeometry* cellGeometry3 = caloGeom->getGeometry (cell3);
    if (cellGeometry3) {
      std::cout << "cell geometry iphi=3 -> ieta=" << ieta
		<< " eta " << cellGeometry3->getPosition().eta () << "+-" 
		<< std::abs(cellGeometry3->getCorners()[0].eta() -
			    cellGeometry3->getCorners()[2].eta())/2
		<< " phi " << cellGeometry3->getPosition().phi ()/3.1415*180 
		<<  "+-" << std::abs(cellGeometry3->getCorners()[0].phi() -
				     cellGeometry3->getCorners()[2].phi())/3.1415*180/2.
		<< std::endl;
    }
  }
}

DEFINE_FWK_MODULE(HcalGeometryTester);
