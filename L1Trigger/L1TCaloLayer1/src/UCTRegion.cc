#include <iostream>
#include <cstdlib>
#include <cstdint>

#include <bitset>
using std::bitset;
#include <string>
using std::string;

#include "UCTRegion.hh"

#include "UCTGeometry.hh"
#include "UCTLogging.hh"

#include "UCTTower.hh"

using namespace l1tcalo;

// Activity fraction to determine how active a tower compared to a region is
// To avoid ratio calculation, one can use comparison to bit-shifted RegionET
// (activityLevelShift, %) = (1, 50%), (2, 25%), (3, 12.5%), (4, 6.125%), (5, 3.0625%)
// Cutting any tighter is rather dangerous
// For the moment we use floating point arithmetic

const float activityFraction = 0.125;
const float ecalActivityFraction = 0.25;
const float miscActivityFraction = 0.25;

bool vetoBit(bitset<4> etaPattern, bitset<4> phiPattern) {
  bitset<4> badPattern5(string("0101"));
  bitset<4> badPattern7(string("0111"));
  bitset<4> badPattern9(string("1001"));
  bitset<4> badPattern10(string("1010"));
  bitset<4> badPattern11(string("1011"));
  bitset<4> badPattern13(string("1101"));
  bitset<4> badPattern14(string("1110"));
  bitset<4> badPattern15(string("1111"));

  bool answer = true;

  if (etaPattern != badPattern5 && etaPattern != badPattern7 && etaPattern != badPattern10 &&
      etaPattern != badPattern11 && etaPattern != badPattern13 && etaPattern != badPattern14 &&
      etaPattern != badPattern15 && phiPattern != badPattern5 &&
      //     phiPattern != badPattern7 && phiPattern != badPattern10 &&
      phiPattern != badPattern10 && phiPattern != badPattern11 && phiPattern != badPattern13 &&
      //phiPattern != badPattern14 && phiPattern != badPattern15 &&
      etaPattern != badPattern9 && phiPattern != badPattern9) {
    answer = false;
  }
  return answer;
}

uint32_t getHitTowerLocation(uint32_t* et) {
  uint32_t etSum = et[0] + et[1] + et[2] + et[3];
  uint32_t iEtSum = (et[0] >> 1) +                 // 0.5xet[0]
                    (et[1] >> 1) + et[1] +         // 1.5xet[1]
                    (et[2] >> 1) + (et[2] << 1) +  // 2.5xet[2]
                    (et[3] << 2) - (et[3] >> 1);   // 3.5xet[3]
  uint32_t iAve = 0xDEADBEEF;
  if (iEtSum <= etSum)
    iAve = 0;
  else if (iEtSum <= (etSum << 1))
    iAve = 1;
  else if (iEtSum <= (etSum + (etSum << 1)))
    iAve = 2;
  else
    iAve = 3;
  return iAve;
}

UCTRegion::UCTRegion(uint32_t crt, uint32_t crd, bool ne, uint32_t rgn, int fwv)
    : crate(crt), card(crd), region(rgn), negativeEta(ne), regionSummary(0), fwVersion(fwv) {
  UCTGeometry g;
  uint32_t nEta = g.getNEta(region);
  uint32_t nPhi = g.getNPhi(region);
  towers.clear();
  for (uint32_t iEta = 0; iEta < nEta; iEta++) {
    for (uint32_t iPhi = 0; iPhi < nPhi; iPhi++) {
      towers.push_back(new UCTTower(crate, card, ne, region, iEta, iPhi, fwVersion));
    }
  }
}

UCTRegion::~UCTRegion() {
  for (uint32_t i = 0; i < towers.size(); i++) {
    if (towers[i] != nullptr)
      delete towers[i];
  }
}

const UCTTower* UCTRegion::getTower(uint32_t caloEta, uint32_t caloPhi) const {
  UCTGeometry g;
  uint32_t nPhi = g.getNPhi(region);
  uint32_t iEta = g.getiEta(caloEta);
  uint32_t iPhi = g.getiPhi(caloPhi);
  UCTTower* tower = towers[iEta * nPhi + iPhi];
  return tower;
}

bool UCTRegion::process() {
  // Determine region dimension
  UCTGeometry g;
  uint32_t nEta = g.getNEta(region);
  uint32_t nPhi = g.getNPhi(region);

  // Process towers and calculate total ET for the region
  uint32_t regionET = 0;
  uint32_t regionEcalET = 0;
  for (uint32_t twr = 0; twr < towers.size(); twr++) {
    if (!towers[twr]->process()) {
      LOG_ERROR << "Tower level processing failed. Bailing out :(" << std::endl;
      return false;
    }
    regionET += towers[twr]->et();
    // Calculate regionEcalET
    regionEcalET += towers[twr]->getEcalET();
  }
  if (regionET > RegionETMask) {
    // Region ET can easily saturate, suppress error spam
    // LOG_ERROR << "L1TCaloLayer1::UCTRegion::Pegging RegionET" << std::endl;
    regionET = RegionETMask;
  }
  regionSummary = (RegionETMask & regionET);
  if (regionEcalET > RegionETMask)
    regionEcalET = RegionETMask;

  // For central regions determine extra bits

  if (region < NRegionsInCard) {
    // Identify active towers
    // Tower ET must be a decent fraction of RegionET
    bool activeTower[nEta][nPhi];
    uint32_t activityLevel = ((uint32_t)((float)regionET) * activityFraction);
    uint32_t activeTowerET = 0;
    for (uint32_t iPhi = 0; iPhi < nPhi; iPhi++) {
      for (uint32_t iEta = 0; iEta < nEta; iEta++) {
        uint32_t towerET = towers[iEta * nPhi + iPhi]->et();
        if (towerET > activityLevel) {
          activeTower[iEta][iPhi] = true;
          activeTowerET += towers[iEta * nPhi + iPhi]->et();
        } else
          activeTower[iEta][iPhi] = false;
      }
    }
    if (activeTowerET > RegionETMask)
      activeTowerET = RegionETMask;
    // Determine "hit" tower as weighted position of ET
    uint32_t sumETIEta[4] = {0, 0, 0, 0};
    for (uint32_t iEta = 0; iEta < nEta; iEta++) {
      for (uint32_t iPhi = 0; iPhi < nPhi; iPhi++) {
        uint32_t towerET = towers[iEta * nPhi + iPhi]->et();
        sumETIEta[iEta] += towerET;
      }
    }
    uint32_t hitIEta = getHitTowerLocation(sumETIEta);
    uint32_t sumETIPhi[4] = {0, 0, 0, 0};
    for (uint32_t iPhi = 0; iPhi < nPhi; iPhi++) {
      for (uint32_t iEta = 0; iEta < nEta; iEta++) {
        uint32_t towerET = towers[iEta * nPhi + iPhi]->et();
        sumETIPhi[iPhi] += towerET;
      }
    }
    uint32_t hitIPhi = getHitTowerLocation(sumETIPhi);
    uint32_t hitTowerLocation = hitIEta * nPhi + hitIPhi;
    // Calculate (energy deposition) active tower pattern
    bitset<4> activeTowerEtaPattern = 0;
    for (uint32_t iEta = 0; iEta < nEta; iEta++) {
      bool activeStrip = false;
      for (uint32_t iPhi = 0; iPhi < nPhi; iPhi++) {
        if (activeTower[iEta][iPhi])
          activeStrip = true;
      }
      if (activeStrip)
        activeTowerEtaPattern |= (0x1 << iEta);
    }
    bitset<4> activeTowerPhiPattern = 0;
    for (uint32_t iPhi = 0; iPhi < nPhi; iPhi++) {
      bool activeStrip = false;
      for (uint32_t iEta = 0; iEta < nEta; iEta++) {
        if (activeTower[iEta][iPhi])
          activeStrip = true;
      }
      if (activeStrip)
        activeTowerPhiPattern |= (0x1 << iPhi);
    }
    // Calculate veto bits for eg and tau patterns
    bool veto = vetoBit(activeTowerEtaPattern, activeTowerPhiPattern);
    bool egVeto = veto;
    bool tauVeto = veto;
    uint32_t maxMiscActivityLevelForEG = ((uint32_t)((float)regionET) * ecalActivityFraction);
    uint32_t maxMiscActivityLevelForTau = ((uint32_t)((float)regionET) * miscActivityFraction);
    if ((regionET - regionEcalET) > maxMiscActivityLevelForEG)
      egVeto = true;
    if ((regionET - activeTowerET) > maxMiscActivityLevelForTau)
      tauVeto = true;

    if (egVeto)
      regionSummary |= RegionEGVeto;
    if (tauVeto)
      regionSummary |= RegionTauVeto;

    regionSummary |= (hitTowerLocation << LocationShift);

    // Extra bits, not in readout, but implicit from their location in data packet for full location information

    if (negativeEta)
      regionSummary |= NegEtaBit;                // Used top bit for +/- eta-side
    regionSummary |= (region << RegionNoShift);  // Max region number 14, so 4 bits needed
    regionSummary |= (card << CardNoShift);      // Max card number is 6, so 3 bits needed
    regionSummary |= (crate << CrateNoShift);    // Max crate number is 2, so 2 bits needed
  }

  return true;
}

bool UCTRegion::clearEvent() {
  regionSummary = 0;
  for (uint32_t i = 0; i < towers.size(); i++) {
    if (!towers[i]->clearEvent())
      return false;
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
  UCTTower* tower = towers[iEta * nPhi + iPhi];
  return tower->setECALData(ecalFG, ecalET);
}

bool UCTRegion::setHCALData(UCTTowerIndex t, uint32_t hcalFB, uint32_t hcalET) {
  UCTGeometry g;
  uint32_t nPhi = g.getNPhi(region);
  uint32_t absCaloEta = abs(t.first);
  uint32_t absCaloPhi = abs(t.second);
  uint32_t iEta = g.getiEta(absCaloEta);
  uint32_t iPhiStart = g.getiPhi(absCaloPhi);
  if (absCaloEta > 29 && absCaloEta < 40) {
    // Valid data are:
    //    absCaloEta = 30-39, 1 < absCaloPhi <= 72 (every second value)
    for (uint32_t iPhi = iPhiStart; iPhi < iPhiStart + 2; iPhi++) {  // For artificial splitting in half
      UCTTower* tower = towers[iEta * nPhi + iPhi];
      // We divide by 2 in output section, after LUT
      if (!tower->setHFData(hcalFB, hcalET))
        return false;
    }
  } else if (absCaloEta == 40 || absCaloEta == 41) {
    // Valid data are:
    //    absCaloEta = 40,41, 1 < absCaloPhi <= 72 (every fourth value)
    for (uint32_t iPhi = 0; iPhi < 4; iPhi++) {  // For artificial splitting in quarter
      UCTTower* tower = towers[iEta * nPhi + iPhi];
      // We divide by 4 in output section, after LUT
      if (!tower->setHFData(hcalFB, hcalET))
        return false;
    }
  } else {
    uint32_t iPhi = g.getiPhi(absCaloPhi);
    UCTTower* tower = towers[iEta * nPhi + iPhi];
    return tower->setHCALData(hcalFB, hcalET);
  }
  return true;
}

bool UCTRegion::setRegionSummary(uint16_t regionData) {
  // Use when the region collection is available and no direct access to TPGs
  regionSummary = regionData;
  return true;
}

std::ostream& operator<<(std::ostream& os, const UCTRegion& r) {
  if (r.negativeEta)
    os << "UCTRegion Summary for negative eta " << r.region << " HitTower (eta, phi) = (" << std::dec << r.hitCaloEta()
       << ", " << r.hitCaloPhi() << ")"
       << " summary = " << std::hex << r.regionSummary << std::endl;
  else
    os << "UCTRegion Summary for positive eta " << r.region << " HitTower (eta, phi) = (" << std::dec << r.hitCaloEta()
       << ", " << r.hitCaloPhi() << ")"
       << " summary = " << std::hex << r.regionSummary << std::endl;

  return os;
}
