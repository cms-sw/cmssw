#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"

#include <iostream>
#include <cassert>

HcalTrigTowerGeometry::HcalTrigTowerGeometry( const HcalTopology* topology )
    : theTopology( topology ) {
  useShortFibers_=true;
  useHFQuadPhiRings_=true;
  useUpgradeConfigurationHFTowers_=!true;
}

void HcalTrigTowerGeometry::setupHF(bool useShortFibers, bool useQuadRings) {
  useShortFibers_=useShortFibers;
  useHFQuadPhiRings_=useQuadRings;
}

std::vector<HcalTrigTowerDetId> 
HcalTrigTowerGeometry::towerIds(const HcalDetId & cellId) const {

  std::vector<HcalTrigTowerDetId> results;

  int HfTowerPhiSize, shift;
  if ( useUpgradeConfigurationHFTowers_ ) { 
    HfTowerPhiSize = 1;
    shift = 0;
  } else {
    HfTowerPhiSize = 4;
    shift = 1;
  }
  
  if(cellId.subdet() == HcalForward) {
    // short fibers don't count
    if(cellId.depth() == 1 || useShortFibers_) {
      // first do eta
      int hfRing = cellId.ietaAbs();
      int ieta = firstHFTower(); 
      // find the tower that contains this ring
      while(hfRing >= firstHFRingInTower(ieta+1)) {
        ++ieta;
      }
      
      if ( useUpgradeConfigurationHFTowers_ && ieta == 29) {
	ieta = 30; 
      }

      ieta *= cellId.zside();

      // now for phi
      // HF towers are quad, 18 in phi.
      // go two cells per trigger tower.

      int iphi = (((cellId.iphi()+shift)/HfTowerPhiSize) * HfTowerPhiSize + shift)%72; // 71+1 --> 1, 3+5 --> 5
      if (useHFQuadPhiRings_ || cellId.ietaAbs() < theTopology->firstHFQuadPhiRing())
        results.push_back( HcalTrigTowerDetId(ieta, iphi) );
    }
      
  } else {
    // the first twenty rings are one-to-one
    if(cellId.ietaAbs() < theTopology->firstHEDoublePhiRing()) {    
      results.push_back( HcalTrigTowerDetId(cellId.ieta(), cellId.iphi()) );
    } else {
      // the remaining rings are two-to-one in phi
      int iphi1 = cellId.iphi();
      int ieta = cellId.ieta();
      // the last eta ring in HE is split.  Recombine.
      if(ieta == theTopology->lastHERing()) --ieta;
      if(ieta == -theTopology->lastHERing()) ++ieta;

      results.push_back( HcalTrigTowerDetId(ieta, iphi1) );
      results.push_back( HcalTrigTowerDetId(ieta, iphi1+1) );
    }
  }

  return results;
}


std::vector<HcalDetId>
HcalTrigTowerGeometry::detIds(const HcalTrigTowerDetId & hcalTrigTowerDetId) const {
  // Written, tested by E. Berry (Princeton)
  std::vector<HcalDetId> results;

  int tower_ieta = hcalTrigTowerDetId.ieta();
  int tower_iphi = hcalTrigTowerDetId.iphi();

  int cell_ieta = tower_ieta;
  int cell_iphi = tower_iphi;

  int min_depth, n_depths;

  // HB
  
  if (abs(cell_ieta) <= theTopology->lastHBRing()){
    theTopology->depthBinInformation(HcalBarrel, abs(tower_ieta), n_depths, min_depth);
    for (int cell_depth = min_depth; cell_depth <= min_depth + n_depths - 1; cell_depth++)
      results.push_back(HcalDetId(HcalBarrel,cell_ieta,cell_iphi,cell_depth));
  }

  // HO
  
  if (abs(cell_ieta) <= theTopology->lastHORing()){ 
    theTopology->depthBinInformation(HcalOuter , abs(tower_ieta), n_depths, min_depth);  
    for (int ho_depth = min_depth; ho_depth <= min_depth + n_depths - 1; ho_depth++)
      results.push_back(HcalDetId(HcalOuter, cell_ieta,cell_iphi,ho_depth));
  }

  // HE 

  if (abs(cell_ieta) >= theTopology->firstHERing() && 
      abs(cell_ieta) <  theTopology->lastHERing()){   

    theTopology->depthBinInformation(HcalEndcap, abs(tower_ieta), n_depths, min_depth);
    
    // Special for double-phi cells
    if (abs(cell_ieta) >= theTopology->firstHEDoublePhiRing())
      if (tower_iphi%2 == 0) cell_iphi = tower_iphi - 1;
    
    for (int cell_depth = min_depth; cell_depth <= min_depth + n_depths - 1; cell_depth++)
      results.push_back(HcalDetId(HcalEndcap, cell_ieta, cell_iphi, cell_depth));
    
    // Special for split-eta cells
    if (abs(tower_ieta) == 28){
      theTopology->depthBinInformation(HcalEndcap, abs(tower_ieta)+1, n_depths, min_depth);
      for (int cell_depth = min_depth; cell_depth <= min_depth + n_depths - 1; cell_depth++){
	if (tower_ieta < 0) results.push_back(HcalDetId(HcalEndcap, tower_ieta - 1, cell_iphi, cell_depth));
	if (tower_ieta > 0) results.push_back(HcalDetId(HcalEndcap, tower_ieta + 1, cell_iphi, cell_depth));
      }
    }
    
  }
    
  // HF 

  if (abs(cell_ieta) >= theTopology->firstHFRing()){  

    int HfTowerPhiSize;
    if   ( useUpgradeConfigurationHFTowers_ ) HfTowerPhiSize = 1;
    else                                      HfTowerPhiSize = 72 / nPhiBins(tower_ieta);
    
    int HfTowerEtaSize     = hfTowerEtaSize(tower_ieta);
    int FirstHFRingInTower = firstHFRingInTower(abs(tower_ieta));

    for (int iHFTowerPhiSegment = 0; iHFTowerPhiSegment < HfTowerPhiSize; iHFTowerPhiSegment++){      
            
      cell_iphi =  (tower_iphi / HfTowerPhiSize) * HfTowerPhiSize; // Find the minimum phi segment
      if ( !useUpgradeConfigurationHFTowers_ ) cell_iphi -= 2; // The first trigger tower starts at HCAL iphi = 71, not HCAL iphi = 1
      cell_iphi += iHFTowerPhiSegment;       // Get all of the HCAL iphi values in this trigger tower
      cell_iphi += 72;                       // Don't want to take the mod of a negative number
      cell_iphi =  cell_iphi % 72;           // There are, at most, 72 cells.
      if ( !useUpgradeConfigurationHFTowers_) cell_iphi += 1;// There is no cell at iphi = 0
      
      if (cell_iphi%2 == 0) continue;        // These cells don't exist.

      for (int iHFTowerEtaSegment = 0; iHFTowerEtaSegment < HfTowerEtaSize; iHFTowerEtaSegment++){
		
	cell_ieta = FirstHFRingInTower + iHFTowerEtaSegment;

	if (cell_ieta >= 40 && cell_iphi%4 == 1) continue;  // These cells don't exist.

	theTopology->depthBinInformation(HcalForward, cell_ieta, n_depths, min_depth);  

	// Negative tower_ieta -> negative cell_ieta
	int zside = 1;
	if (tower_ieta < 0) zside = -1;

	cell_ieta *= zside;	       

	for (int cell_depth = min_depth; cell_depth <= min_depth + n_depths - 1; cell_depth++)
	  results.push_back(HcalDetId(HcalForward, cell_ieta, cell_iphi, cell_depth));
	
	if ( zside * cell_ieta == 30 ) {
	  theTopology->depthBinInformation(HcalForward, 29 * zside, n_depths, min_depth);  
	  for (int cell_depth = min_depth; cell_depth <= min_depth + n_depths - 1; cell_depth++) 
	    results.push_back(HcalDetId(HcalForward, 29 * zside , cell_iphi, cell_depth));
	}
	
      }    
    }
  }

  return results;
}


int HcalTrigTowerGeometry::hfTowerEtaSize(int ieta) const {
  
  if ( useUpgradeConfigurationHFTowers_ ) return 1;
  
  int ietaAbs = abs(ieta); 
  assert(ietaAbs >= firstHFTower() && ietaAbs <= nTowers());
  // the first three come from rings 29-31, 32-34, 35-37. The last has 4 rings: 38-41
  return (ietaAbs == nTowers()) ? 4 : 3;
  
}


int HcalTrigTowerGeometry::firstHFRingInTower(int ietaTower) const {
  // count up to the correct HF ring
  int inputTower = abs(ietaTower);
  int result = theTopology->firstHFRing();
  for(int iTower = firstHFTower(); iTower != inputTower; ++iTower) {
    result += hfTowerEtaSize(iTower);
  }
  
  // negative in, negative out.
  if(ietaTower < 0) result *= -1;
  return result;
}


void HcalTrigTowerGeometry::towerEtaBounds(int ieta, double & eta1, double & eta2) const {
  int ietaAbs = abs(ieta);
  std::pair<double,double> etas = 
    (ietaAbs < firstHFTower()) ? theTopology->etaRange(HcalBarrel,ietaAbs) : 
    theTopology->etaRange(HcalForward,ietaAbs);
  eta1 = etas.first;
  eta2 = etas.second;
  
  // get the signs and order right
  if(ieta < 0) {
    double tmp = eta1;
    eta1 = -eta2;
    eta2 = -tmp;
  }
}
