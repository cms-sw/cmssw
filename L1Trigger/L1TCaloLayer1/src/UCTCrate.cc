#include <iostream>
#include <stdlib.h>
#include <stdint.h>

#include "UCTCrate.hh"
#include "UCTCard.hh"
#include "UCTGeometry.hh"
#include "UCTLogging.hh"

UCTCrate::UCTCrate(uint32_t crt) :
  crate(crt),
  crateSummary(0) {
  UCTGeometry g;
  for(uint32_t card = 0; card < g.getNCards(); card++) {
    cards.push_back(new UCTCard(crate, card));
  }
}

UCTCrate::~UCTCrate() {
  for(uint32_t i = 0; i < cards.size(); i++) {
    if(cards[i] != 0) delete cards[i];
  }
}

bool UCTCrate::process() {
  crateSummary = 0;
  for(uint32_t i = 0; i < cards.size(); i++) {
    if(cards[i] != 0) {
      cards[i]->process();
      crateSummary += cards[i]->et();
    }
  }
  return true;
}

bool UCTCrate::clearEvent() {
  crateSummary = 0;
  for(uint32_t i = 0; i < cards.size(); i++) {
    if(!cards[i]->clearEvent()) return false;
  }
  return true;
}

bool UCTCrate::setECALData(UCTTowerIndex t, bool ecalFG, uint32_t ecalET) {
  UCTGeometry g;
  uint32_t i = g.getCard(t.first, t.second);
  if(i > cards.size()) {
    LOG_ERROR << "UCTCrate: Incorrect (caloEta, caloPhi) -- bailing" << std::endl;
    exit(1);
  }
  return cards[i]->setECALData(t, ecalFG, ecalET);
}

bool UCTCrate::setHCALData(UCTTowerIndex t, uint32_t hcalFB, uint32_t hcalET) {
  UCTGeometry g;
  uint32_t i = g.getCard(t.first, t.second);
  if(i > cards.size()) {
    LOG_ERROR << "UCTCrate: Incorrect (caloEta, caloPhi) -- bailing" << std::endl;
    exit(1);
  }
  return cards[i]->setHCALData(t, hcalFB, hcalET);
}

const UCTCard* UCTCrate::getCard(UCTTowerIndex t) const {
  UCTGeometry g;
  uint32_t i = g.getCard(t.first, t.second);
  if(i > cards.size()) {
    LOG_ERROR << "UCTCrate: Incorrect (caloEta, caloPhi) -- bailing" << std::endl;
    exit(1);
  }
  return cards[i];
}

std::ostream& operator<<(std::ostream& os, const UCTCrate& cr) {
  if(cr.crateSummary > 0) 
    os << "UCTCrate: crate = " << cr.crate << "; Summary = " << cr.crateSummary << std::endl;
  return os;
}
