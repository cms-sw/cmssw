#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include "Geometry/HcalTowerAlgo/src/HcalHardcodeGeometryData.h"

#include <iostream>

HcalTrigTowerGeometry::HcalTrigTowerGeometry() {
  useShortFibers_=true;
  useHFQuadPhiRings_=true;
}

void HcalTrigTowerGeometry::setupHF(bool useShortFibers, bool useQuadRings) {
  useShortFibers_=useShortFibers;
  useHFQuadPhiRings_=useQuadRings;
}

std::vector<HcalTrigTowerDetId> 
HcalTrigTowerGeometry::towerIds(const HcalDetId & cellId) const {

  std::vector<HcalTrigTowerDetId> results;

  if(cellId.subdet() == HcalForward) {
    // short fibers don't count
    if(cellId.depth() == 1 || useShortFibers_) {
      // first do eta
      int hfRing = cellId.ietaAbs();
      int ieta = firstHFTower(); 
      // find the tower that contains this ring
      while(hfRing > firstHFRingInTower(ieta+1)) {
        ++ieta;
      }

      ieta *= cellId.zside();

      // now for phi
      // HF towers are quad, 18 in phi.  
      // go two cells per trigger tower.  
      int iphi = cellId.iphi();
      if(cellId.ietaAbs() < theTopology.firstHFQuadPhiRing()) { 
	iphi = (((iphi+1)/4)* 4 + 1)%72; // 71+1 --> 1, 3+5 --> 5
      }
      if (useHFQuadPhiRings_ || cellId.ietaAbs() < theTopology.firstHFQuadPhiRing())
        results.push_back( HcalTrigTowerDetId(ieta, iphi) );
    }
      
  } else {
    // the first twenty rings are one-to-one
    if(cellId.ietaAbs() < theTopology.firstHEDoublePhiRing()) {    
      results.push_back( HcalTrigTowerDetId(cellId.ieta(), cellId.iphi()) );
    } else {
      // the remaining rings are two-to-one in phi
      int iphi1 = (cellId.iphi()-1)*2 + 1;
      int ieta = cellId.ieta();
      // the last eta ring in HE is split.  Recombine.
      if(ieta == theTopology.lastHERing()) --ieta;

      results.push_back( HcalTrigTowerDetId(ieta, iphi1) );
      results.push_back( HcalTrigTowerDetId(ieta, iphi1+1) );
    }
  }

  return results;
}


std::vector<HcalDetId> 
HcalTrigTowerGeometry::detIds(const HcalTrigTowerDetId &) const {
  std::vector<HcalDetId> results;
  return results;
}


int HcalTrigTowerGeometry::hfTowerEtaSize(int ieta) const {
  int ietaAbs = abs(ieta); 
  assert(ietaAbs >= firstHFTower());
  // the first comes from rings 29-32.  The rest have 3 rings each
  return (ietaAbs == firstHFTower()) ? 4 : 3;
}


int HcalTrigTowerGeometry::firstHFRingInTower(int ietaTower) const {
  // count up to the correct HF ring
  int inputTower = abs(ietaTower);
  int result = theTopology.firstHFRing();
  for(int iTower = firstHFTower(); iTower != inputTower; ++iTower) {
    result += hfTowerEtaSize(iTower);
  }
  
  // negative in, negative out.
  if(ietaTower < 0) result *= -1;
  return result;
}


void HcalTrigTowerGeometry::towerEtaBounds(int ieta, double & eta1, double & eta2) const {
  int ietaAbs = abs(ieta);
  if(ietaAbs < firstHFTower()) {
    eta1 = theHBHEEtaBounds[ietaAbs-1];
    eta2 = theHBHEEtaBounds[ieta];
    // the last tower is split, so get tower 29, too
    if(ieta == theTopology.lastHERing()-1) {
      eta2 = theHBHEEtaBounds[ieta+1];
    } 
  } else {
    // count from 0
    int hfIndex = firstHFRingInTower(ietaAbs) - theTopology.firstHFRing();
    eta1 = theHFEtaBounds[hfIndex];
    eta2 = theHFEtaBounds[hfIndex+ hfTowerEtaSize(ieta)];
  }

  // get the signs right
  if(ieta < 0) eta1 *= -1;
  if(ieta < 0) eta2 *= -1;
}

