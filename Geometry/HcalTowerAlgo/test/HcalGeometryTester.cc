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
  ~HcalGeometryTester( void ) {}
    
  void beginJob() override {}
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endJob() override {}

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
  edm::LogInfo("HcalGeometryTester") << "Test SLHC Hcal Flexi geometry";
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

  std::stringstream s;
  s << label << " : " << std::endl;
  const std::vector<DetId>& idshb = caloGeom->getValidDetIds(det, subdet);
 
  int counter = 0;
  for (std::vector<DetId>::const_iterator i=idshb.begin(); i!=idshb.end(); 
       i++, ++counter) {
    HcalDetId hid=(*i);
    s << counter << ": din " << topology.detId2denseId(*i) << ":" 
	      << hid;
    const CaloCellGeometry * cell = caloGeom->getGeometry(*i);
    s << *cell << std::endl;
  }
 
  s << "=== Total " << counter << " cells in " << label << std::endl;
  edm::LogInfo("HcalGeometryTester") << s.str();
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
  HcalDetId closest = geom->getClosestCell(cell->getPosition());
  edm::LogInfo("HcalGeometryTester") << "i/p " << detId << " position " << cell->getPosition() << " closest " << closest;

  if(closest != detId) {
    edm::LogError("HcalGeometryTester") << "Mismatch.  Original HCAL cell is "
	      << detId << " while nearest is " << closest;
  }
}

void HcalGeometryTester::testTriggerGeometry(const HcalTopology& topology) {

  HcalTrigTowerGeometry trigTowers( &topology );
  edm::LogInfo("HcalGeometryTester") << "HCAL trigger tower eta bounds ";
  for(int ieta = 1; ieta <= 32; ++ieta) {
    double eta1, eta2;
    trigTowers.towerEtaBounds(ieta, 0, eta1, eta2);
    edm::LogInfo("HcalGeometryTester") << ieta << " "  << eta1 << " " << eta2;
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
    edm::LogInfo("HcalGeometryTester") << "Trigger Tower Size: Barrel " << barrelTowers.size();
    assert(barrelTowers.size() ==1);
    edm::LogInfo("HcalGeometryTester") << barrelTowers[0];
  } 
  if (topology.valid(endcapDet)) {
    TowerDets endcapTowers = trigTowers.towerIds(endcapDet);
    edm::LogInfo("HcalGeometryTester") << "Trigger Tower Size: Endcap " << endcapTowers.size();
    assert(endcapTowers.size() >=1);
    edm::LogInfo("HcalGeometryTester") << endcapTowers[0];
    if (endcapTowers.size() > 1)
      edm::LogInfo("HcalGeometryTester") << endcapTowers[1];
  }
  if (topology.valid(forwardDet1)) {
    TowerDets forwardTowers1 = trigTowers.towerIds(forwardDet1);
    edm::LogInfo("HcalGeometryTester") << "Trigger Tower Size: Forward1 " << forwardTowers1.size();
    assert(forwardTowers1.size() ==1);
    edm::LogInfo("HcalGeometryTester") << forwardTowers1[0];
  }
  if (topology.valid(forwardDet1)) {
    TowerDets forwardTowers2 = trigTowers.towerIds(forwardDet2);
    edm::LogInfo("HcalGeometryTester") << "Trigger Tower Size: Forward2 " << forwardTowers2.size();
    assert(forwardTowers2.size() ==1);
    edm::LogInfo("HcalGeometryTester") << forwardTowers2[0];
  }
  if (topology.valid(forwardDet1)) {
    TowerDets forwardTowers3 = trigTowers.towerIds(forwardDet3);
    edm::LogInfo("HcalGeometryTester") << "Trigger Tower Size: Forward3 " << forwardTowers3.size();
    assert(forwardTowers3.size() ==1);
    edm::LogInfo("HcalGeometryTester") << forwardTowers3[0];
  }
}

void HcalGeometryTester::testFlexiValidDetIds(CaloSubdetectorGeometry* caloGeom,
					      const HcalTopology& topology, 
					      DetId::Detector det, int subdet, 
					      std::string label, 
					      std::vector<int> &dins) {

  std::stringstream s;
  s << label << " : " << std::endl;
  const std::vector<DetId>& idshb = caloGeom->getValidDetIds(det, subdet);
    
  int counter = 0;
  for (std::vector<DetId>::const_iterator i=idshb.begin(); i!=idshb.end(); 
       i++, ++counter) {
    HcalDetId hid=(*i);
    s << counter << ": din " << topology.detId2denseId(*i) << ":" << hid;
    dins.push_back( topology.detId2denseId(*i));
	
    const CaloCellGeometry * cell = caloGeom->getGeometry(*i);
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
  edm::LogInfo("HcalGeometryTester") << s.str();
}

void HcalGeometryTester::testFlexiGeomHF(CaloSubdetectorGeometry* caloGeom) {

  std::stringstream s;
  s << "Test HF Geometry : " << std::endl;
  for (int ieta = 29; ieta <=41; ++ieta) {
    HcalDetId cell3 (HcalForward, ieta, 3, 1);
    const CaloCellGeometry* cellGeometry3 = caloGeom->getGeometry (cell3);
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
  edm::LogInfo("HcalGeometryTester") << s.str();
}

DEFINE_FWK_MODULE(HcalGeometryTester);
