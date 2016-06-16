#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <stdint.h>

#include "UCTLayer1.hh"

#include "UCTCrate.hh"
#include "UCTCard.hh"
#include "UCTRegion.hh"
#include "UCTTower.hh"

#include "UCTGeometry.hh"
#include "UCTLogging.hh"

using namespace l1tcalo;

UCTLayer1::UCTLayer1() : uctSummary(0) {
  UCTGeometry g;
  crates.reserve(g.getNCrates());
  for(uint32_t crate = 0; crate < g.getNCrates(); crate++) {
    crates.push_back(new UCTCrate(crate));
  }
}

UCTLayer1::~UCTLayer1() {
  for(uint32_t i = 0; i < crates.size(); i++) {
    if(crates[i] != 0) delete crates[i];
  }
}

bool UCTLayer1::clearEvent() {
  for(uint32_t i = 0; i < crates.size(); i++) {
    if(crates[i] != 0) crates[i]->clearEvent();
  }
  return true;
}

const UCTRegion* UCTLayer1::getRegion(int regionEtaIndex, uint32_t regionPhiIndex) const {
  if(regionEtaIndex == 0 || (uint32_t) std::abs(regionEtaIndex) > NRegionsInCard || regionPhiIndex >= MaxUCTRegionsPhi) {
    return 0;
  }
  // Get (0,0) tower region information
  UCTGeometry g;
  UCTRegionIndex r = UCTRegionIndex(regionEtaIndex, regionPhiIndex);
  UCTTowerIndex t = g.getUCTTowerIndex(r);
  uint32_t absCaloEta = abs(t.first);
  uint32_t absCaloPhi = abs(t.second);
  uint32_t crt = g.getCrate(absCaloEta, absCaloPhi);
  if(crt >= crates.size()) {
    LOG_ERROR << "UCTLayer1::getRegion - Crate number is wrong - " << std::hex << crt 
	      << std::dec
	      << " (rEta,rPhi)=(" << regionEtaIndex << ","<< regionPhiIndex << ")" 
	      << " (eta,phi)=(" << absCaloEta << ","<< absCaloPhi << ")" << std::endl;
    exit(1);
  }
  const UCTCrate* crate = crates[crt];
  const UCTCard* card = crate->getCard(t);
  const UCTRegion* region = card->getRegion(r);
  return region;
}

const UCTTower* UCTLayer1::getTower(int caloEta, int caloPhi) const {
  if(caloPhi < 0) {
    LOG_ERROR << "UCT::getTower - Negative caloPhi is unacceptable -- bailing" << std::endl;
    exit(1);
  }
  UCTGeometry g;
  UCTTowerIndex twr = UCTTowerIndex(caloEta, caloPhi);
  const UCTRegionIndex rgn = g.getUCTRegionIndex(twr);
  const UCTRegion* region = getRegion(rgn);
  const UCTTower* tower = region->getTower(twr);
  return tower;
}

bool UCTLayer1::setECALData(UCTTowerIndex t, bool ecalFG, uint32_t ecalET) {
  uint32_t absCaloEta = abs(t.first);
  uint32_t absCaloPhi = abs(t.second);
  UCTGeometry g;
  uint32_t crt = g.getCrate(absCaloEta, absCaloPhi);
  if(crt >= crates.size()) {
    LOG_ERROR << "UCTLayer1::setECALData - Crate number is wrong - " << std::hex << crt << std::dec
	      << " (eta,phi)=(" << absCaloEta << ","<< absCaloPhi << ")" << std::endl;
    exit(1);
  }
  UCTCrate* crate = crates[crt];
  return crate->setECALData(t, ecalFG, ecalET);
}

bool UCTLayer1::setHCALData(UCTTowerIndex t, uint32_t hcalFB, uint32_t hcalET) {
  uint32_t absCaloEta = abs(t.first);
  uint32_t absCaloPhi = abs(t.second);
  UCTGeometry g;
  uint32_t crt = g.getCrate(absCaloEta, absCaloPhi);
  if(crt >= crates.size()) {
    LOG_ERROR << "UCTLayer1::setHCALData - Crate number is wrong - " << std::hex << crt << std::dec
	      << " (eta,phi)=(" << absCaloEta << ","<< absCaloPhi << ")" << std::endl;
    exit(1);
  }
  UCTCrate* crate = crates[crt];
  return crate->setHCALData(t, hcalFB, hcalET);
}

bool UCTLayer1::process() {
  uctSummary = 0;
  for(uint32_t i = 0; i < crates.size(); i++) {
    if(crates[i] != 0) {
      crates[i]->process();
      uctSummary += crates[i]->et();
    }
  }

  return true;
}

std::ostream& operator<<(std::ostream& os, const UCTLayer1& l) {
  os << "UCTLayer1: Summary " << l.uctSummary << std::endl;
  return os;
}

