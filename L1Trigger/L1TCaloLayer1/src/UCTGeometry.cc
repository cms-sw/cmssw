#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <array>

#include "UCTGeometry.hh"
#include "UCTLogging.hh"
using namespace l1tcalo;

UCTGeometry::UCTGeometry() {}

uint32_t UCTGeometry::getLinkNumber(bool negativeEta, uint32_t region, uint32_t iEta, uint32_t iPhi) {
  if (checkRegion(region)) {
    LOG_ERROR << "Invalid region number: region = " << region << std::endl;
    exit(1);
  }
  if (checkEtaIndex(region, iEta)) {
    LOG_ERROR << "Invalid eta index: iEta = " << iEta << std::endl;
    exit(1);
  }
  if (checkPhiIndex(region, iPhi)) {
    LOG_ERROR << "Invalid eta index: iPhi = " << iPhi << std::endl;
    exit(1);
  }
  uint32_t linkNumber = 0xDEADBEEF;
  if (region < MaxRegionNumber) {
    if (iEta < NEtaInRegion / 2) {
      linkNumber = region * 2;
    } else {
      linkNumber = region * 2 + 1;
    }
  } else {
    linkNumber = NRegionsInCard * 2 + iPhi;
  }

  if (!negativeEta) {
    linkNumber += NRegionsInCard * 2 + 2;
  }
  return linkNumber;
}

int UCTGeometry::getCaloEtaIndex(bool negativeSide, uint32_t region, uint32_t iEta) {
  if (checkRegion(region)) {
    LOG_ERROR << "Invalid region number: region = " << region << std::endl;
    exit(1);
  }
  if (checkEtaIndex(region, iEta)) {
    LOG_ERROR << "Invalid eta index: iEta = " << iEta << std::endl;
    exit(1);
  }

  int caloEtaIndex = region * NEtaInRegion + iEta + 1;
  if (region > 6) {
    caloEtaIndex = (region - 7) * NHFEtaInRegion + iEta + 30;
  }

  if (negativeSide)
    return -caloEtaIndex;
  return caloEtaIndex;
}

int UCTGeometry::getCaloPhiIndex(uint32_t crate, uint32_t card, uint32_t region, uint32_t iPhi) {
  if (checkCrate(crate)) {
    LOG_ERROR << "Invalid crate number: crate = " << crate << std::endl;
    exit(1);
  }
  if (checkCard(card)) {
    LOG_ERROR << "Invalid card number: card = " << card << std::endl;
    exit(1);
  }
  if (checkPhiIndex(region, iPhi)) {
    LOG_ERROR << "Invalid phi index: iPhi = " << iPhi << std::endl;
    exit(1);
  }
  int caloPhiIndex = 0xDEADBEEF;
  if (crate == 0) {
    caloPhiIndex = 11 + card * 4 + iPhi;
  } else if (crate == 1) {
    caloPhiIndex = 59 + card * 4 + iPhi;
  } else if (crate == 2) {
    caloPhiIndex = 35 + card * 4 + iPhi;
  }
  if (caloPhiIndex > 72)
    caloPhiIndex -= 72;
  return caloPhiIndex;
}

uint32_t UCTGeometry::getUCTRegionPhiIndex(uint32_t crate, uint32_t card) {
  if (checkCrate(crate)) {
    LOG_ERROR << "Invalid crate number: crate = " << crate << std::endl;
    exit(1);
  }
  if (checkCard(card)) {
    LOG_ERROR << "Invalid card number: card = " << card << std::endl;
    exit(1);
  }
  uint32_t uctRegionPhiIndex = 0xDEADBEEF;
  if (crate == 0) {
    uctRegionPhiIndex = 3 + card;
  } else if (crate == 1) {
    if (card < 3) {
      uctRegionPhiIndex = 15 + card;
    } else {
      uctRegionPhiIndex = card - 3;
    }
  } else if (crate == 2) {
    uctRegionPhiIndex = 9 + card;
  }
  return uctRegionPhiIndex;
}

uint32_t UCTGeometry::getCrate(int caloEta, int caloPhi) {
  uint32_t crate = 0xDEADBEEF;
  if (caloPhi >= 11 && caloPhi <= 34)
    crate = 0;
  else if (caloPhi >= 35 && caloPhi <= 58)
    crate = 2;
  else if (caloPhi >= 59 && caloPhi <= 72)
    crate = 1;
  else if (caloPhi >= 1 && caloPhi <= 10)
    crate = 1;
  return crate;
}

uint32_t UCTGeometry::getCard(int caloEta, int caloPhi) {
  uint32_t crate = getCrate(caloEta, caloPhi);
  uint32_t card = 0xDEADBEEF;
  if (crate == 0) {
    card = (caloPhi - 11) / 4;
  } else if (crate == 2) {
    card = (caloPhi - 35) / 4;
  } else if (crate == 1 && caloPhi > 58) {
    card = (caloPhi - 59) / 4;
  } else if (crate == 1 && caloPhi <= 10) {
    card = (caloPhi + 13) / 4;
  }
  return card;
}

uint32_t UCTGeometry::getRegion(int caloEta, int caloPhi) {
  uint32_t absCEta = std::abs(caloEta);
  if ((absCEta - 1) < (NRegionsInCard * NEtaInRegion))
    return (absCEta - 1) / NEtaInRegion;
  else
    return NRegionsInCard + ((absCEta - 2 - (NRegionsInCard * NEtaInRegion)) / NHFEtaInRegion);
}

uint32_t UCTGeometry::getiEta(int caloEta) {
  uint32_t absCEta = std::abs(caloEta);
  if ((absCEta - 1) < (NRegionsInCard * NEtaInRegion))
    return (absCEta - 1) % NEtaInRegion;
  else
    return absCEta % NHFEtaInRegion;  // To account for missing tower 29
}

uint32_t UCTGeometry::getiPhi(int caloPhi) { return (caloPhi + 1) % NPhiInCard; }

uint32_t UCTGeometry::getNEta(uint32_t region) {
  uint32_t nEta = 0xDEADBEEF;
  if (region < CaloHFRegionStart) {
    nEta = NEtaInRegion;
  } else {
    nEta = NHFEtaInRegion;
  }
  return nEta;
}

uint32_t UCTGeometry::getNPhi(uint32_t region) { return NPhiInRegion; }

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
  if (iPhi >= NPhiInRegion || iEta >= NEtaInRegion) {
    return UCTTowerIndex(0, 0);  // Illegal values
  }
  int regionEta = region.first;
  uint32_t regionPhi = region.second;
  bool negativeSide = (regionEta < 0);
  uint32_t regionNo = std::abs(regionEta) - 1;
  int towerEta = 0xDEADBEEF;
  if (regionNo < NRegionsInCard) {
    towerEta = 1 + regionNo * getNEta(regionNo) + iEta;  // Ranges 1 - 28
  } else if (regionNo < (NRegionsInCard + NHFRegionsInCard)) {
    towerEta = HFEtaOffset + 1 + (regionNo - NRegionsInCard) * getNEta(regionNo) + iEta;  // Ranges 30 - 42
  }
  if (negativeSide)
    towerEta = -towerEta;
  int towerPhi = regionPhi * NPhiInRegion + iPhi - 1;  // Always pretend that there are 4 phi-regions
  if (towerPhi <= 0)
    towerPhi += 72;
  // Legal values of towerPhi = 1, 2, 5, 6, ..., 69, 70 for towerEta 30-39
  // Legal values of towerPhi = 1, 5, ..., 69 for towerEta 40-41
  return UCTTowerIndex(towerEta, towerPhi);
}

namespace {
  constexpr std::array<double, 42> fillTwrEtaValues() {
    std::array<double, 42> twrEtaValues = {{0}};
    twrEtaValues[0] = 0;
    for (unsigned int i = 0; i < 20; i++) {
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
    twrEtaValues[29] = 999.;
    twrEtaValues[30] = (3.15 + 2.98) / 2.;
    twrEtaValues[31] = (3.33 + 3.15) / 2.;
    twrEtaValues[32] = (3.50 + 3.33) / 2.;
    twrEtaValues[33] = (3.68 + 3.50) / 2.;
    twrEtaValues[34] = (3.68 + 3.85) / 2.;
    twrEtaValues[35] = (3.85 + 4.03) / 2.;
    twrEtaValues[36] = (4.03 + 4.20) / 2.;
    twrEtaValues[37] = (4.20 + 4.38) / 2.;
    twrEtaValues[38] = (4.74 + 4.38 * 3) / 4.;
    twrEtaValues[39] = (4.38 + 4.74 * 3) / 4.;
    twrEtaValues[40] = (5.21 + 4.74 * 3) / 4.;
    twrEtaValues[41] = (4.74 + 5.21 * 3) / 4.;
    return twrEtaValues;
  }
  constexpr std::array<double, 42> twrEtaValues = fillTwrEtaValues();
}  // namespace

double UCTGeometry::getUCTTowerEta(int caloEta) {
  uint32_t absCaloEta = std::abs(caloEta);
  if (absCaloEta <= 41) {
    if (caloEta < 0)
      return -twrEtaValues[absCaloEta];
    else
      return +twrEtaValues[absCaloEta];
  } else
    return -999.;
}

double UCTGeometry::getUCTTowerPhi(int caloPhi) {
  if (caloPhi < 1)
    return -999.;
  else if (caloPhi > 72)
    return +999.;
  uint32_t absCaloPhi = std::abs(caloPhi) - 1;
  if (absCaloPhi < 36)
    return (((double)absCaloPhi + 0.5) * 0.0872);
  else
    return (-(71.5 - (double)absCaloPhi) * 0.0872);
}

UCTRegionIndex UCTGeometry::getUCTRegionIndexFromL1CaloRegion(uint32_t caloRegionEta, uint32_t caloRegionPhi) {
  uint32_t region = 0xDEADBEEF;
  bool negativeEtaSide = false;
  if (caloRegionEta == 31) {
    region = 12;
    negativeEtaSide = true;
  } else if (caloRegionEta == 30) {
    region = 11;
    negativeEtaSide = true;
  } else if (caloRegionEta <= 10) {
    region = 10 - caloRegionEta;
    negativeEtaSide = true;
  } else if (caloRegionEta >= 11 && caloRegionEta <= 23) {
    region = caloRegionEta - 11;
  }
  return UCTRegionIndex(getUCTRegionEtaIndex(negativeEtaSide, region), caloRegionPhi);
}

UCTTowerIndex UCTGeometry::getUCTTowerIndexFromL1CaloRegion(UCTRegionIndex r, uint32_t rawData) {
  uint32_t iEta = (rawData >> 14) & 0x3;
  uint32_t iPhi = (rawData >> 12) & 0x3;
  return getUCTTowerIndex(r, iEta, iPhi);
}
