#include <iostream>
#include <stdlib.h>
#include <stdint.h>

#include <string>
using std::string;

#include "UCTRegion.hh"

#include "UCTGeometry.hh"

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
  uint32_t nEta = g.getNEta(region);
  uint32_t iEta = g.getiEta(caloEta, caloPhi);
  uint32_t iPhi = g.getiPhi(caloEta, caloPhi);
  UCTTower* tower = towers[iEta*nEta+iPhi];
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
      std::cerr << "Tower level processing failed. Bailing out :(" << std::endl;
      return false;
    }
    regionET += towers[twr]->et();
  }
  if(regionET > RegionETMask) {
    std::cerr << std::dec;
    std::cerr << "towers.size() = " << towers.size() << std::endl;
    for(uint32_t twr = 0; twr < towers.size(); twr++) {
      std::cerr << "towers[twr]->et() = " << towers[twr]->et() << std::endl;
    }
    std::cerr << "Pegging RegionET" << std::endl;
    regionET = RegionETMask;
  }
  regionSummary = (RegionETMask & regionET);

  // For central regions determine extra bits

  if(region < NRegionsInCard) {
    uint32_t highestTowerET = 0;
    uint32_t highestTowerLocation = 0;
    for(uint32_t iPhi = 0; iPhi < nPhi; iPhi++) {
      for(uint32_t iEta = 0; iEta < nEta; iEta++) {
	uint32_t towerET = towers[iEta*nEta+iPhi]->et();
	if(highestTowerET < towerET ) {
	  highestTowerET = towerET;
	  highestTowerLocation = iEta*nEta+iPhi;
	}
      }
    }
    regionSummary |= (highestTowerLocation << LocationShift);
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
  uint32_t nEta = g.getNEta(region);
  uint32_t absCaloEta = abs(t.first);
  uint32_t absCaloPhi = abs(t.second);
  uint32_t iEta = g.getiEta(absCaloEta, absCaloPhi);
  uint32_t iPhi = g.getiPhi(absCaloEta, absCaloPhi);
  UCTTower* tower = towers[iEta*nEta+iPhi];
  return tower->setECALData(ecalFG, ecalET);
}

bool UCTRegion::setHCALData(UCTTowerIndex t, uint32_t hcalFB, uint32_t hcalET) {
  UCTGeometry g;
  uint32_t nEta = g.getNEta(region);
  uint32_t absCaloEta = abs(t.first);
  uint32_t absCaloPhi = abs(t.second);
  uint32_t iEta = g.getiEta(absCaloEta, absCaloPhi);
  if(absCaloEta > 29 && absCaloEta < 40) {
    uint32_t iPhi = (absCaloPhi % 2) * 2; // iPhi == 0 or 2
    UCTTower* tower = towers[iEta*nEta + iPhi]; // iPhi == 0 or 2
    if(!tower->setHFData(hcalFB, (hcalET - hcalET / 2))) return false;
    tower = towers[iEta*nEta + iPhi + 1]; // iPhi + 1 == 1 or 3
    if(!tower->setHFData(hcalFB, hcalET / 2)) return false;
  }
  else if(absCaloEta == 40 || absCaloEta == 41) {
    for(uint32_t iPhi = 0; iPhi < 4; iPhi++) { // For artificial splitting in quads
      UCTTower* tower = towers[iEta*nEta + iPhi];
      if(!tower->setHFData(hcalFB, hcalET / 4)) return false;
    }
  }
  else {
    uint32_t iPhi = g.getiPhi(absCaloEta, absCaloPhi);
    UCTTower* tower = towers[iEta*nEta+iPhi];
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
