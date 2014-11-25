#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalHardcodeGeometryLoader.h"
#include "Geometry/HcalTowerAlgo/interface/HcalFlexiHardcodeGeometryLoader.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"
#include <memory>
#include <string>
#include <iostream>

void testTriggerGeometry( HcalTopology& topology ) {

  HcalTrigTowerGeometry trigTowers( &topology );

  std::cout << "HCAL trigger tower eta bounds (version 0)" << std::endl;
  for(int ieta = 1; ieta <= 32; ++ieta) {
    double eta1, eta2;
    trigTowers.towerEtaBounds(ieta, 0, eta1, eta2);
    std::cout << ieta << " "  << eta1 << " " << eta2 << std::endl;
  }

  std::cout << "HCAL trigger tower eta bounds (version 1)" << std::endl;
  for(int ieta = 1; ieta <= 41; ++ieta) {
    double eta1, eta2;
    trigTowers.towerEtaBounds(ieta, 1, eta1, eta2);
    std::cout << ieta << " "  << eta1 << " " << eta2 << std::endl;
  }

  // now test some cell mappings
  HcalDetId barrelDet(HcalBarrel, 1, 1, 1);
  HcalDetId endcapDet(HcalEndcap, 29, 1, 1);
  HcalDetId forwardDet1(HcalForward, 29, 71, 1);
  HcalDetId forwardDet2(HcalForward, 29, 71, 2);
  HcalDetId forwardDet3(HcalForward, 40, 71, 1);

  typedef std::vector<HcalTrigTowerDetId> TowerDets;
  TowerDets barrelTowers = trigTowers.towerIds(barrelDet);
  TowerDets endcapTowers = trigTowers.towerIds(endcapDet);
  TowerDets forwardTowers1 = trigTowers.towerIds(forwardDet1);
  TowerDets forwardTowers2 = trigTowers.towerIds(forwardDet2);
  TowerDets forwardTowers3 = trigTowers.towerIds(forwardDet3);

  // Each version of the HF Trigger Towers will add an additional item to the
  // vector returned by trigTowers.towerIds() for forward towers.
  size_t foward_det_count = 0;
  if (trigTowers.useRCT()) { foward_det_count++; }
  if (trigTowers.use1x1()) { foward_det_count++; }

  assert(barrelTowers.size() ==1);
  assert(endcapTowers.size() ==2);
  assert(forwardTowers1.size() == foward_det_count);
  assert(forwardTowers2.size() == foward_det_count);
  assert(forwardTowers3.size() == foward_det_count);

  std::cout << barrelTowers[0] << std::endl;
  std::cout << endcapTowers[0] << std::endl;
  std::cout << endcapTowers[1] << std::endl;
  std::cout << forwardTowers1[0] << std::endl;
  std::cout << forwardTowers3[0] << std::endl;

}

void
testClosestCell(const HcalDetId & detId, const CaloSubdetectorGeometry * geom)
{
  const CaloCellGeometry* cell = geom->getGeometry(detId);
  std::cout << "i/p " << detId << " position " << cell->getPosition();
  HcalDetId closest = geom->getClosestCell( cell->getPosition() );
  std::cout << " closest " << closest << std::endl;

  if(closest != detId) {
    std::cout << "ERROR mismatch.  Original HCAL cell is "
	      << detId << " while nearest is " << closest << std::endl;
  }
}

void
testClosestCells( HcalTopology& topology ) 
{
  edm::FileInPath fp("Geometry/HcalTowerAlgo/test/runHcalGeometryAnalyzer_cfg.py");
  std::string fname  = fp.fullPath();
  std::shared_ptr<edm::ParameterSet> pS = edm::readConfig(fname);
  //  std::cout << pS->dump() << std::endl;

  edm::ParameterSet const& pModule =
    pS->getParameter<edm::ParameterSet>("HcalHardcodeGeometryEP@");
  HcalFlexiHardcodeGeometryLoader loader(pModule);

  CaloSubdetectorGeometry* g = loader.load( topology );
  
  // make sure each cel is its own closest cell
  HcalDetId barrelDet(HcalBarrel, 1, 1, 1);
  HcalDetId barrelDet2(HcalBarrel, 16, 50, 1);
  HcalDetId endcapDet1(HcalEndcap, -17, 72, 1);
  HcalDetId endcapDet2(HcalEndcap, 29, 35, 1);
  HcalDetId forwardDet1(HcalForward, 30, 71, 1);
  HcalDetId forwardDet3(HcalForward, -40, 71, 1);

  testClosestCell( barrelDet  , g );
  testClosestCell( barrelDet2 , g );
  testClosestCell( endcapDet1 , g );
  testClosestCell( endcapDet2 , g );
  testClosestCell( forwardDet1, g );
  testClosestCell( forwardDet3, g );
    
  const std::vector<DetId>& ids=g->getValidDetIds(DetId::Hcal,HcalBarrel);
  for (std::vector<DetId>::const_iterator i=ids.begin(); i!=ids.end(); i++) {
    testClosestCell( HcalDetId(*i), g );
  }
}

void
testValidDetIds( HcalTopology& topology, DetId::Detector det, int subdet, char const* label )
{
  HcalHardcodeGeometryLoader loader( topology );
  HcalHardcodeGeometryLoader::ReturnType caloGeom = loader.load( det, subdet );
  std::cout << std::endl << label << " : " << std::endl;
  const std::vector<DetId>& idshb = caloGeom->getValidDetIds( det, subdet );
  
  int counter = 0;
  for (std::vector<DetId>::const_iterator i=idshb.begin(); i!=idshb.end(); i++, ++counter) {
    HcalDetId hid=(*i);
    std::cout << counter << ": din " << topology.detId2denseId(*i) << ":" << hid;
    const CaloCellGeometry * cell = caloGeom->getGeometry(*i);
    std::cout << *cell << std::endl;
  }

  std::cout << "=== Total " << counter << " cells in " << label << std::endl;
}

void
testFlexiValidDetIds( HcalTopology& topology, DetId::Detector det, int subdet, char const* label, std::vector<int> &dins )
{

  edm::FileInPath fp("Geometry/HcalTowerAlgo/test/runHcalGeometryAnalyzer_cfg.py");
  std::string fname  = fp.fullPath();
  std::shared_ptr<edm::ParameterSet> pS = edm::readConfig(fname);
  //  std::cout << pS->dump() << std::endl;

  edm::ParameterSet const& pModule =
    pS->getParameter<edm::ParameterSet>("HcalHardcodeGeometryEP@");
  HcalFlexiHardcodeGeometryLoader loader(pModule);

  CaloSubdetectorGeometry* caloGeom = loader.load( topology );
  std::cout << std::endl << label << " : " << std::endl;
  const std::vector<DetId>& idshb = caloGeom->getValidDetIds( det, subdet );

  //std::vector<int> dins;
    
  int counter = 0;
  for (std::vector<DetId>::const_iterator i=idshb.begin(); i!=idshb.end(); i++, ++counter) {
    HcalDetId hid=(*i);
    std::cout << counter << ": din " << topology.detId2denseId(*i) << ":" << hid;
    dins.push_back( topology.detId2denseId(*i));
	
    const CaloCellGeometry * cell = caloGeom->getGeometry(*i);
    std::cout << *cell << std::endl;
  }

  std::sort( dins.begin(), dins.end());
  std::cout << "=== Total " << counter << " cells in " << label << std::endl;

  counter = 0;
  for (std::vector<int>::const_iterator i=dins.begin(); i != dins.end(); ++i, ++counter) {
    HcalDetId hid = (topology.denseId2detId(*i));
    HcalDetId ihid = (topology.denseId2detId(dins[counter]));
    std::cout << counter << ": din " << (*i) << " :" << hid << " == " << ihid << std::endl;
  }
}

int main() {

  std::cout << "Test current Hcal geometry" << std::endl;

  HcalTopologyMode::Mode mode = HcalTopologyMode::LHC;
  int maxDepthHB = 2;
  int maxDepthHE = 3;
  HcalTopology topology( mode, maxDepthHB, maxDepthHE );

  testValidDetIds( topology, DetId::Hcal, HcalBarrel, " BARREL " );
  testValidDetIds( topology, DetId::Hcal, HcalEndcap, " ENDCAP " );
  testValidDetIds( topology, DetId::Hcal, HcalOuter, " OUTER " );
  testValidDetIds( topology, DetId::Hcal, HcalForward, " FORWARD " );

  testTriggerGeometry( topology );

  testClosestCells( topology );

  std::cout << "Test SLHC Hcal geometry" << std::endl;
    
  mode = HcalTopologyMode::SLHC;
  maxDepthHB = 7;
  maxDepthHE = 7;
  HcalTopology stopology( mode, maxDepthHB, maxDepthHE );

  testValidDetIds( stopology, DetId::Hcal, HcalBarrel, " SLHC BARREL " );
  testValidDetIds( stopology, DetId::Hcal, HcalEndcap, " SLHC ENDCAP " );
  testValidDetIds( stopology, DetId::Hcal, HcalOuter, " SLHC OUTER " );
  testValidDetIds( stopology, DetId::Hcal, HcalForward, " SLHC FORWARD " );

  std::cout << "Test SLHC Hcal Flexi geometry" << std::endl;
  std::vector<int> dins;

  testFlexiValidDetIds( stopology, DetId::Hcal, HcalBarrel, " SLHC BARREL ", dins );
  testFlexiValidDetIds( stopology, DetId::Hcal, HcalEndcap, " SLHC ENDCAP ", dins );
  testFlexiValidDetIds( stopology, DetId::Hcal, HcalOuter, " SLHC OUTER ", dins );
  testFlexiValidDetIds( stopology, DetId::Hcal, HcalForward, " SLHC FORWARD ", dins );
    
  testTriggerGeometry( stopology );

  testClosestCells( stopology );
    
  return 0;
}
