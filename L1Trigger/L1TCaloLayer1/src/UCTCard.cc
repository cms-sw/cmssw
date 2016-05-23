#include <iostream>
#include <stdlib.h>
#include <stdint.h>

#include "UCTCard.hh"
#include "UCTRegion.hh"
#include "UCTGeometry.hh"
#include "UCTLogging.hh"

UCTCard::UCTCard(uint32_t crt, uint32_t crd) :
  crate(crt),
  card(crd),
  cardSummary(0) {
  UCTGeometry g;
  regions.reserve(2*g.getNRegions());
  for(uint32_t rgn = 0; rgn < g.getNRegions(); rgn++) {
    // Negative eta side
    regions.push_back(new UCTRegion(crate, card, true, rgn));
    // Positive eta side
    regions.push_back(new UCTRegion(crate, card, false, rgn));
  }
}

UCTCard::~UCTCard() {
  for(uint32_t i = 0; i < regions.size(); i++) {
    if(regions[i] != 0) delete regions[i];
  }
}

bool UCTCard::process() {
  cardSummary = 0;
  for(uint32_t i = 0; i < regions.size(); i++) {
    if(regions[i] != 0) regions[i]->process();
    cardSummary += regions[i]->et();
  }
  return true;
}

bool UCTCard::clearEvent() {
  cardSummary = 0;
  for(uint32_t i = 0; i < regions.size(); i++) {
    if(!regions[i]->clearEvent()) return false;
  }
  return true;
}

bool UCTCard::setECALData(UCTTowerIndex t, bool ecalFG, uint32_t ecalET) {
  UCTGeometry g;
  uint32_t absCaloEta = abs(t.first);
  uint32_t absCaloPhi = abs(t.second);
  bool negativeEta = false;
  if(t.first < 0) negativeEta = true;
  uint32_t i = g.getRegion(absCaloEta, absCaloPhi) * 2;
  if(!negativeEta) i++;
  if(i > regions.size()) {
    LOG_ERROR << "UCTCard: Incorrect region requested -- bailing" << std::endl;
    exit(1);
  }
  return regions[i]->setECALData(t, ecalFG, ecalET);
}

bool UCTCard::setHCALData(UCTTowerIndex t, uint32_t hcalFB, uint32_t hcalET) {
  UCTGeometry g;
  uint32_t absCaloEta = abs(t.first);
  uint32_t absCaloPhi = abs(t.second);
  bool negativeEta = false;
  if(t.first < 0) negativeEta = true;
  uint32_t i = g.getRegion(absCaloEta, absCaloPhi) * 2;
  if(!negativeEta) i++;
  if(i > regions.size()) {
    LOG_ERROR << "UCTCard: Incorrect region requested -- bailing" << std::endl;
    exit(1);
  }
  return regions[i]->setHCALData(t, hcalFB, hcalET);
}

const UCTRegion* UCTCard::getRegion(UCTRegionIndex r) const {
  UCTGeometry g;
  UCTTowerIndex t = g.getUCTTowerIndex(r);
  uint32_t absCaloEta = std::abs(t.first);
  uint32_t absCaloPhi = std::abs(t.second);
  bool negativeEta = false;
  if(t.first < 0) negativeEta = true;
  return getRegion(negativeEta, absCaloEta, absCaloPhi);
}

const UCTRegion* UCTCard::getRegion(bool nE, uint32_t cEta, uint32_t cPhi) const {
  UCTGeometry g;
  uint32_t i = g.getRegion(cEta, cPhi) * 2;
  if(!nE) i++;
  if(i > regions.size()) {
    LOG_ERROR << "UCTCard: Incorrect region requested -- bailing" << std::endl;
    exit(1);
  }
  return regions[i];
}

std::ostream& operator<<(std::ostream& os, const UCTCard& c) {
  if(c.cardSummary > 0)
    os << "UCTCard: card = " << c.card << "; Summary = " << c.cardSummary << std::endl;
  return os;
}
