#include <iostream>
#include <stdlib.h>
#include <stdint.h>

#include "UCTGeometry.hh"
#include "UCTLogging.hh"

UCTGeometry::UCTGeometry() {
  twrEtaValues[0] = 0;
  for(unsigned int i = 0; i < 20; i++) {
    twrEtaValues[i + 1] = 0.0436 + i * 0.0872;
  }
  twrEtaValues[21] = 1.785;
  twrEtaValues[22] = 1.880;
  twrEtaValues[23] = 1.9865;
  twrEtaValues[24] = 2.1075;
  twrEtaValues[25] = 2.247;
  twrEtaValues[26] = 2.411;
  twrEtaValues[27] = 2.575;
  twrEtaValues[28] = 2.825;
}

uint32_t UCTGeometry::getLinkNumber(bool negativeEta, uint32_t region, 
				    uint32_t iEta, uint32_t iPhi) {
  if(checkRegion(region)) {
    LOG_ERROR << "Invalid region number: region = " << region << std::endl;
    exit(1);
  }
  if(checkEtaIndex(region, iEta)) {
    LOG_ERROR << "Invalid eta index: iEta = " << iEta << std::endl;
    exit(1);
  }
  if(checkPhiIndex(region, iPhi)) {
    LOG_ERROR << "Invalid eta index: iPhi = " << iPhi << std::endl;
    exit(1);
  }
  uint32_t linkNumber = 0xDEADBEEF;
  if(region < l1tcalo::MaxRegionNumber) {
    if(iEta < l1tcalo::NEtaInRegion / 2) {
      linkNumber = region * 2;
    }
    else {
      linkNumber = region * 2 + 1;
    }
  }
  else {
    linkNumber = l1tcalo::NRegionsInCard * 2 + iPhi;
  }

  if(!negativeEta) {
    linkNumber += l1tcalo::NRegionsInCard * 2 + 2;
  }
  return linkNumber;
}

int UCTGeometry::getCaloEtaIndex(bool negativeSide, uint32_t region, uint32_t iEta) {

  if(checkRegion(region)) {
    LOG_ERROR << "Invalid region number: region = " << region << std::endl;
    exit(1);
  }
  if(checkEtaIndex(region, iEta)) {
    LOG_ERROR << "Invalid eta index: iEta = " << iEta << std::endl;
    exit(1);
  }

  int caloEtaIndex = region * l1tcalo::NEtaInRegion + iEta + 1;
  if(region > 6) {
    caloEtaIndex = (region - 7) * l1tcalo::NHFEtaInRegion + iEta + 30;
  }

  if(negativeSide) return -caloEtaIndex;
  return caloEtaIndex;

}

int UCTGeometry::getCaloPhiIndex(uint32_t crate, uint32_t card, 
				 uint32_t region, uint32_t iPhi) {
  if(checkCrate(crate)) {
    LOG_ERROR << "Invalid crate number: crate = " << crate << std::endl;
    exit(1);
  }
  if(checkCard(card)) {
    LOG_ERROR << "Invalid card number: card = " << card << std::endl;
    exit(1);
  }
  if(checkPhiIndex(region, iPhi)) {
    LOG_ERROR << "Invalid phi index: iPhi = " << iPhi << std::endl;
    exit(1);
  }
  int caloPhiIndex = 0xDEADBEEF;
  if(crate == 0) {
    caloPhiIndex = 11 + card * 4 + iPhi;
  }
  else if(crate == 1) {
    caloPhiIndex = 59 + card * 4 + iPhi;
  }
  else if(crate == 2) {
    caloPhiIndex = 35 + card * 4 + iPhi;
  }
  if(caloPhiIndex > 72) caloPhiIndex -= 72;
  return caloPhiIndex;
}

uint32_t UCTGeometry::getUCTRegionPhiIndex(uint32_t crate, uint32_t card) {
  if(checkCrate(crate)) {
    LOG_ERROR << "Invalid crate number: crate = " << crate << std::endl;
    exit(1);
  }
  if(checkCard(card)) {
    LOG_ERROR << "Invalid card number: card = " << card << std::endl;
    exit(1);
  }
  uint32_t uctRegionPhiIndex = 0xDEADBEEF;
  if(crate == 0) {
    uctRegionPhiIndex = 3 + card;
  }
  else if(crate == 1) {
    if(card < 3) {
      uctRegionPhiIndex = 15 + card;
    }
    else {
      uctRegionPhiIndex = card - 3;
    }
  }
  else if(crate == 2) {
    uctRegionPhiIndex = 9 + card;
  }
  return uctRegionPhiIndex;
}

uint32_t UCTGeometry::getCrate(int caloEta, int caloPhi) {
  uint32_t crate = 0xDEADBEEF;
  if(caloPhi >= 11 && caloPhi <= 34) crate = 0;
  else if(caloPhi >= 35 && caloPhi <= 58) crate = 2;
  else if(caloPhi >= 59 && caloPhi <= 72) crate = 1;
  else if(caloPhi >= 1 && caloPhi <= 10) crate = 1;  
  return crate;
}

uint32_t UCTGeometry::getCard(int caloEta, int caloPhi) {
  uint32_t crate = getCrate(caloEta, caloPhi);
  uint32_t card = 0xDEADBEEF;
  if(crate == 0) {
    card = (caloPhi - 11) / 4;
  }
  else if(crate == 2) {
    card = (caloPhi - 35) / 4;
  }
  else if(crate == 1 && caloPhi > 58) {
    card = (caloPhi - 59) / 4;
  }
  else if(crate == 1 && caloPhi <= 10) {
    card = (caloPhi + 13) / 4;
  }    
  return card;
}

uint32_t UCTGeometry::getRegion(int caloEta, int caloPhi) {
  uint32_t absCEta = abs(caloEta);
  if((absCEta - 1) < (l1tcalo::NRegionsInCard * l1tcalo::NEtaInRegion))
    return (absCEta - 1) / l1tcalo::NEtaInRegion;
  else
    return l1tcalo::NRegionsInCard + ((absCEta - 2 - (l1tcalo::NRegionsInCard * l1tcalo::NEtaInRegion)) / l1tcalo::NHFEtaInRegion);
}

uint32_t UCTGeometry::getiEta(int caloEta) {
  uint32_t absCEta = abs(caloEta);
  if((absCEta - 1) < (l1tcalo::NRegionsInCard * l1tcalo::NEtaInRegion))
    return (absCEta - 1) % l1tcalo::NEtaInRegion;
  else
    return absCEta % l1tcalo::NHFEtaInRegion;  // To account for missing tower 29
}

uint32_t UCTGeometry::getiPhi(int caloPhi) {
  return (caloPhi + 1) % l1tcalo::NPhiInCard;
}

uint32_t UCTGeometry::getNEta(uint32_t region) {
  uint32_t nEta = 0xDEADBEEF;
  if(region < l1tcalo::CaloHFRegionStart) {
    nEta = l1tcalo::NEtaInRegion;
  }
  else {
    nEta = l1tcalo::NHFEtaInRegion;
  }
  return nEta;
}

uint32_t UCTGeometry::getNPhi(uint32_t region) {
  return l1tcalo::NPhiInRegion;
}

UCTRegionIndex UCTGeometry::getUCTRegionIndex(int caloEta, int caloPhi) {
  uint32_t regionPhi = getUCTRegionPhiIndex(getCrate(caloEta, caloPhi), getCard(caloEta, caloPhi));
  int regionEta = getUCTRegionEtaIndex((caloEta < 0), getRegion(caloEta, caloPhi));
  return UCTRegionIndex(regionEta, regionPhi);
}

UCTRegionIndex UCTGeometry::getUCTRegionIndex(bool negativeSide, uint32_t crate, uint32_t card, uint32_t region) {
  uint32_t regionPhi = getUCTRegionPhiIndex(crate, card);
  int regionEta = getUCTRegionEtaIndex(negativeSide, region);
  return UCTRegionIndex(regionEta, regionPhi);
}

UCTTowerIndex UCTGeometry::getUCTTowerIndex(UCTRegionIndex region, uint32_t iEta, uint32_t iPhi) {
  if(iPhi >= l1tcalo::NPhiInRegion || iEta >= l1tcalo::NEtaInRegion) {
    return UCTTowerIndex(0, 0); // Illegal values
  }
  int regionEta = region.first;
  int absRegionEta = abs(regionEta);
  int towerEta = (regionEta / absRegionEta) * (absRegionEta * getNEta(absRegionEta) + iEta);
  uint32_t regionPhi = region.second;
  int towerPhi = regionPhi * getNPhi(absRegionEta) + iPhi;
  return UCTTowerIndex(towerEta, towerPhi);
}

double UCTGeometry::getUCTTowerEta(int caloEta) {
  uint32_t absCaloEta = abs(caloEta);
  if(absCaloEta <= 28) return twrEtaValues[absCaloEta];
  else return -999.;
}

double UCTGeometry::getUCTTowerPhi(int caloPhi, int caloEta) {
  if(caloPhi < 0) return -999.;
  uint32_t absCaloPhi = abs(caloPhi) - 1;
  return (((double) absCaloPhi + 0.5) * 0.0872);
}
