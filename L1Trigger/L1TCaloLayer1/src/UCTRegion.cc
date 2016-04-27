#include <iostream>
#include <stdlib.h>
#include <stdint.h>

#include <string>
using std::string;

#include "UCTRegion.hh"

#include "UCTGeometry.hh"
#include "UCTLogging.hh"

#include "UCTTower.hh"

UCTRegion::UCTRegion(uint32_t crt, uint32_t crd, bool ne, uint32_t rgn) :
  crate(crt),
  card(crd),
  region(rgn),
  negativeEta(ne),
  regionSummary(0) {
  UCTGeometry g;
  uint32_t nEta = g.getNEta(region);
  uint32_t nPhi = g.getNPhi(region);
  towers.clear();
  for(uint32_t iEta = 0; iEta < nEta; iEta++) {
    for(uint32_t iPhi = 0; iPhi < nPhi; iPhi++) {
      towers.push_back(new UCTTower(crate, card, ne, region, iEta, iPhi));
    }
  }
}

UCTRegion::~UCTRegion() {
  for(uint32_t i = 0; i < towers.size(); i++) {
    if(towers[i] != 0) delete towers[i];
  }
}

const UCTTower* UCTRegion::getTower(uint32_t caloEta, uint32_t caloPhi) const {
  UCTGeometry g;
  uint32_t nPhi = g.getNPhi(region);
  uint32_t iEta = g.getiEta(caloEta);
  uint32_t iPhi = g.getiPhi(caloPhi);
  UCTTower* tower = towers[iEta*nPhi+iPhi];
  return tower;
}

bool UCTRegion::process() {

  // Determine region dimension
  UCTGeometry g;
  uint32_t nEta = g.getNEta(region);
  uint32_t nPhi = g.getNPhi(region);

  // Process towers and calculate total ET for the region
  uint32_t regionET = 0;
  for(uint32_t twr = 0; twr < towers.size(); twr++) {
    if(!towers[twr]->process()) {
      LOG_ERROR << "Tower level processing failed. Bailing out :(" << std::endl;
      return false;
    }
    regionET += towers[twr]->et();
  }
  if(regionET > l1tcalo::RegionETMask) {
    LOG_ERROR << "L1TCaloLayer1::UCTRegion::Pegging RegionET" << std::endl;
    regionET = l1tcalo::RegionETMask;
  }
  regionSummary = (l1tcalo::RegionETMask & regionET);

  // For central regions determine extra bits

  if(region < l1tcalo::NRegionsInCard) {
    uint32_t highestTowerET = 0;
    uint32_t highestTowerLocation = 0;
    for(uint32_t iPhi = 0; iPhi < nPhi; iPhi++) {
      for(uint32_t iEta = 0; iEta < nEta; iEta++) {
	uint32_t towerET = towers[iEta*nPhi+iPhi]->et();
	if(highestTowerET < towerET ) {
	  highestTowerET = towerET;
	  highestTowerLocation = iEta*nPhi+iPhi;
	}
      }
    }
    regionSummary |= (highestTowerLocation << l1tcalo::LocationShift);
  }
  
  return true;

}

bool UCTRegion::clearEvent() {
  regionSummary = 0;
  for(uint32_t i = 0; i < towers.size(); i++) {
    if(!towers[i]->clearEvent()) return false;
  }
  return true;
}

bool UCTRegion::setECALData(UCTTowerIndex t, bool ecalFG, uint32_t ecalET) {
  UCTGeometry g;
  uint32_t nPhi = g.getNPhi(region);
  uint32_t absCaloEta = abs(t.first);
  uint32_t absCaloPhi = abs(t.second);
  uint32_t iEta = g.getiEta(absCaloEta);
  uint32_t iPhi = g.getiPhi(absCaloPhi);
  UCTTower* tower = towers[iEta*nPhi+iPhi];
  return tower->setECALData(ecalFG, ecalET);
}

bool UCTRegion::setHCALData(UCTTowerIndex t, uint32_t hcalFB, uint32_t hcalET) {
  UCTGeometry g;
  uint32_t nPhi = g.getNPhi(region);
  uint32_t absCaloEta = abs(t.first);
  uint32_t absCaloPhi = abs(t.second);
  uint32_t iEta = g.getiEta(absCaloEta);
  uint32_t iPhiStart = g.getiPhi(absCaloPhi);
  if(absCaloEta > 29 && absCaloEta < 40) {
    // Valid data are: 
    //    absCaloEta = 30-39, 1 < absCaloPhi <= 72 (every second value)
    for(uint32_t iPhi = iPhiStart; iPhi < iPhiStart + 2; iPhi++) { // For artificial splitting in half
      UCTTower* tower = towers[iEta*nPhi + iPhi];
      // We divide by 2 in output section, after LUT
      if(!tower->setHFData(hcalFB, hcalET)) return false;
    }
  }
  else if(absCaloEta == 40 || absCaloEta == 41) {
    // Valid data are: 
    //    absCaloEta = 40,41, 1 < absCaloPhi <= 72 (every fourth value)
    for(uint32_t iPhi = 0; iPhi < 4; iPhi++) { // For artificial splitting in quarter
      UCTTower* tower = towers[iEta * nPhi + iPhi];
      // We divide by 4 in output section, after LUT
      if(!tower->setHFData(hcalFB, hcalET)) return false;
    }
  }
  else {
    uint32_t iPhi = g.getiPhi(absCaloPhi);
    UCTTower* tower = towers[iEta*nPhi+iPhi];
    return tower->setHCALData(hcalFB, hcalET);
  }
  return true;
}

std::ostream& operator<<(std::ostream& os, const UCTRegion& r) {
  if(r.negativeEta)
    os << "UCTRegion Summary for negative eta " << r.region << " summary = "<< std:: hex << r.regionSummary << std::endl;
  else
    os << "UCTRegion Summary for positive eta " << r.region << " summary = "<< std:: hex << r.regionSummary << std::endl;

  return os;
}
