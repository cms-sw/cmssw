#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"

#include <iostream>
#include <cassert>

HcalTrigTowerGeometry::HcalTrigTowerGeometry( const HcalTopology* topology )
   : theTopology(topology)
{
   auto tmode = theTopology->triggerMode();
   useRCT_ = tmode <= HcalTopologyMode::TriggerMode_2016;
   use1x1_ = tmode >= HcalTopologyMode::TriggerMode_2016;
   use2017_ = tmode >= HcalTopologyMode::TriggerMode_2017 or
              tmode == HcalTopologyMode::TriggerMode_2018legacy;
}

std::vector<HcalTrigTowerDetId> 
HcalTrigTowerGeometry::towerIds(const HcalDetId & cellId) const {

  std::vector<HcalTrigTowerDetId> results;

  if(cellId.subdet() == HcalForward) {

    if (useRCT_) { 
      // first do eta
      int hfRing = cellId.ietaAbs();
      int ieta = firstHFTower(0); 
      // find the tower that contains this ring
      while(hfRing >= firstHFRingInTower(ieta+1)) {
	++ieta;
      }
      
      ieta *= cellId.zside();
      
      // now for phi
      // HF towers are quad, 18 in phi.
      // go two cells per trigger tower.
      
      int iphi = (((cellId.iphi()+1)/4) * 4 + 1)%72; // 71+1 --> 1, 3+5 --> 5
      results.emplace_back( HcalTrigTowerDetId(ieta, iphi) );
    } 
    if (use1x1_) {
      int hfRing = cellId.ietaAbs();
      if (hfRing==29) hfRing=30; // sum 29 into 30.

      int ieta = hfRing*cellId.zside();      
      int iphi = cellId.iphi();

      HcalTrigTowerDetId id(ieta,iphi);
      id.setVersion(1); // version 1 for 1x1 HF granularity
      results.emplace_back(id);
    }
      
  } else {
    // the first twenty rings are one-to-one
    if(cellId.ietaAbs() < theTopology->firstHEDoublePhiRing()) {    
      results.emplace_back( HcalTrigTowerDetId(cellId.ieta(), cellId.iphi()) );
    } else {
      // the remaining rings are two-to-one in phi
      int iphi1 = cellId.iphi();
      int ieta = cellId.ieta();
      int depth = cellId.depth();
      // the last eta ring in HE is split.  Recombine.
      if(ieta == theTopology->lastHERing()) --ieta;
      if(ieta == -theTopology->lastHERing()) ++ieta;

      if (use2017_) {
         if (ieta == 26 and depth == 7)
            ++ieta;
         if (ieta == -26 and depth == 7)
            --ieta;
      }

      results.emplace_back( HcalTrigTowerDetId(ieta, iphi1) );
      results.emplace_back( HcalTrigTowerDetId(ieta, iphi1+1) );
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
    theTopology->depthBinInformation(HcalBarrel, abs(tower_ieta), tower_iphi, hcalTrigTowerDetId.zside(), n_depths, min_depth);
    for (int cell_depth = min_depth; cell_depth <= min_depth + n_depths - 1; cell_depth++)
      results.emplace_back(HcalDetId(HcalBarrel,cell_ieta,cell_iphi,cell_depth));
  }

  // HO
  
  if (abs(cell_ieta) <= theTopology->lastHORing()){ 
    theTopology->depthBinInformation(HcalOuter , abs(tower_ieta), tower_iphi, hcalTrigTowerDetId.zside(), n_depths, min_depth);  
    for (int ho_depth = min_depth; ho_depth <= min_depth + n_depths - 1; ho_depth++)
      results.emplace_back(HcalDetId(HcalOuter, cell_ieta,cell_iphi,ho_depth));
  }

  // HE 

  if (abs(cell_ieta) >= theTopology->firstHERing() && 
      abs(cell_ieta) <  theTopology->lastHERing()){   

    theTopology->depthBinInformation(HcalEndcap, abs(tower_ieta), tower_iphi, hcalTrigTowerDetId.zside(), n_depths, min_depth);
    
    // Special for double-phi cells
    if (abs(cell_ieta) >= theTopology->firstHEDoublePhiRing())
      if (tower_iphi%2 == 0) cell_iphi = tower_iphi - 1;

    if (use2017_) {
         if (abs(tower_ieta) == 26)
            --n_depths;
         if (tower_ieta == 27)
            results.emplace_back(HcalDetId(HcalEndcap, cell_ieta - 1, cell_iphi, 7));
         if (tower_ieta == -27)
            results.emplace_back(HcalDetId(HcalEndcap, cell_ieta + 1, cell_iphi, 7));
    }
    
    for (int cell_depth = min_depth; cell_depth <= min_depth + n_depths - 1; cell_depth++)
      results.emplace_back(HcalDetId(HcalEndcap, cell_ieta, cell_iphi, cell_depth));
    
    // Special for split-eta cells
    if (abs(tower_ieta) == 28){
      theTopology->depthBinInformation(HcalEndcap, abs(tower_ieta)+1, tower_iphi, hcalTrigTowerDetId.zside(), n_depths, min_depth);
      for (int cell_depth = min_depth; cell_depth <= min_depth + n_depths - 1; cell_depth++){
	if (tower_ieta < 0) results.emplace_back(HcalDetId(HcalEndcap, tower_ieta - 1, cell_iphi, cell_depth));
	if (tower_ieta > 0) results.emplace_back(HcalDetId(HcalEndcap, tower_ieta + 1, cell_iphi, cell_depth));
      }
    }
    
  }
    
  // HF 

  if (abs(cell_ieta) >= theTopology->firstHFRing()){  

    if (hcalTrigTowerDetId.version()==0) {
    
      int HfTowerPhiSize =  72 / nPhiBins(tower_ieta,0);

      int HfTowerEtaSize     = hfTowerEtaSize(tower_ieta);
      int FirstHFRingInTower = firstHFRingInTower(abs(tower_ieta));

      for (int iHFTowerPhiSegment = 0; iHFTowerPhiSegment < HfTowerPhiSize; iHFTowerPhiSegment++){      
            
	cell_iphi =  (tower_iphi / HfTowerPhiSize) * HfTowerPhiSize; // Find the minimum phi segment
	cell_iphi -= 2; // The first trigger tower starts at HCAL iphi = 71, not HCAL iphi = 1
	cell_iphi += iHFTowerPhiSegment;       // Get all of the HCAL iphi values in this trigger tower
	cell_iphi += 72;                       // Don't want to take the mod of a negative number
	cell_iphi =  cell_iphi % 72;           // There are, at most, 72 cells.
	cell_iphi += 1;// There is no cell at iphi = 0
      
	if (cell_iphi%2 == 0) continue;        // These cells don't exist.

	for (int iHFTowerEtaSegment = 0; iHFTowerEtaSegment < HfTowerEtaSize; iHFTowerEtaSegment++){
		
	  cell_ieta = FirstHFRingInTower + iHFTowerEtaSegment;

	  if (cell_ieta >= 40 && cell_iphi%4 == 1) continue;  // These cells don't exist.

	  theTopology->depthBinInformation(HcalForward, cell_ieta, tower_iphi, hcalTrigTowerDetId.zside(), n_depths, min_depth);  
	  
	  // Negative tower_ieta -> negative cell_ieta
	  int zside = 1;
	  if (tower_ieta < 0) zside = -1;

	  cell_ieta *= zside;	       

	  for (int cell_depth = min_depth; cell_depth <= min_depth + n_depths - 1; cell_depth++)
	    results.emplace_back(HcalDetId(HcalForward, cell_ieta, cell_iphi, cell_depth));
	
	  if ( zside * cell_ieta == 30 ) {
	    theTopology->depthBinInformation(HcalForward, 29 * zside, tower_iphi, hcalTrigTowerDetId.zside(), n_depths, min_depth);  
	    for (int cell_depth = min_depth; cell_depth <= min_depth + n_depths - 1; cell_depth++) 
	      results.emplace_back(HcalDetId(HcalForward, 29 * zside , cell_iphi, cell_depth));
	  }
	}
      }  
    } else if (hcalTrigTowerDetId.version()==1) {
      theTopology->depthBinInformation(HcalForward, tower_ieta, tower_iphi, hcalTrigTowerDetId.zside(), n_depths, min_depth);  
      for (int cell_depth = min_depth; cell_depth <= min_depth + n_depths - 1; cell_depth++)
	results.emplace_back(HcalDetId(HcalForward, tower_ieta, tower_iphi, cell_depth));      
      if (abs(tower_ieta)==30) {
	int i29 = 29;
	  if (tower_ieta < 0) i29 = -29;
	  theTopology->depthBinInformation(HcalForward, i29, tower_iphi, hcalTrigTowerDetId.zside(), n_depths, min_depth);  
	  for (int cell_depth = min_depth; cell_depth <= min_depth + n_depths - 1; cell_depth++)
	    results.emplace_back(HcalDetId(HcalForward, i29, tower_iphi, cell_depth));      
      }
    }
  }
  
  return results;
}


int HcalTrigTowerGeometry::hfTowerEtaSize(int ieta) const {
  
  int ietaAbs = abs(ieta); 
  assert(ietaAbs >= firstHFTower(0) && ietaAbs <= nTowers(0));
  // the first three come from rings 29-31, 32-34, 35-37. The last has 4 rings: 38-41
  return (ietaAbs == nTowers(0)) ? 4 : 3;
  
}


int HcalTrigTowerGeometry::firstHFRingInTower(int ietaTower) const {
  // count up to the correct HF ring
  int inputTower = abs(ietaTower);
  int result = theTopology->firstHFRing();
  for(int iTower = firstHFTower(0); iTower != inputTower; ++iTower) {
    result += hfTowerEtaSize(iTower);
  }
  
  // negative in, negative out.
  if(ietaTower < 0) result *= -1;
  return result;
}


void HcalTrigTowerGeometry::towerEtaBounds(int ieta, int version, double & eta1, double & eta2) const {
  int ietaAbs = abs(ieta);
  std::pair<double,double> etas = 
    (ietaAbs < firstHFTower(version)) ? theTopology->etaRange(HcalBarrel,ietaAbs) : 
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
