#include <algorithm>
#include <cmath>

#include "L1TriggerScouting/Utilities/interface/conversion.h"

bool l1ScoutingRun3::calol1::validHwEta(int16_t hwEta) {
  auto const hwEtaAbs = std::abs(hwEta);
  return (hwEtaAbs >= kHwEtaAbsMin and hwEtaAbs <= kHwEtaAbsMax and hwEtaAbs != kHwEtaAbsHFFirst);
}

float l1ScoutingRun3::calol1::fEta(int16_t hwEta) {
  //
  // The array kTowerEtas in this function holds the average (midpoint) of
  // the pseudorapidity boundaries of the calorimeter towers used in the level-1 trigger.
  // These values are based on those listed in CMS NOTE 2005/016 (Table 1),
  // and used in the function l1t::CaloTools::towerEta() implemented in
  //   L1Trigger/L1TCalorimeter/src/CaloTools.cc
  //
  // Note: the midpoint of tower 28 in this function is based on the pseudorapidity boundaries
  // given in CMS NOTE 2005/016 for that tower (i.e. 2.650 < |eta| < 3.000),
  // while in l1t::CaloTools::towerEta() the outer edge of tower 28
  // corresponds to the start of the first HF tower (i.e. 2.650 < |eta| < 2.853).
  //
  // Three non-physical values are used in kTowerEtas at indices 0, 29, and 42.
  // These correspond to |ieta| values that L1T calorimeter towers are never supposed to have
  // (with 42 corresponding to any |ieta| value higher than 41).
  //
  static constexpr float kTowerEtas[kHwEtaAbsMax + 2] = {
      9999.0f, 0.0435f, 0.1305f, 0.2175f, 0.3045f, 0.3915f, 0.4785f, 0.5655f, 0.6525f, 0.7395f, 0.8265f,
      0.9135f, 1.0005f, 1.0875f, 1.1745f, 1.2615f, 1.3485f, 1.4355f, 1.5225f, 1.6095f, 1.6965f, 1.7850f,
      1.8800f, 1.9865f, 2.1075f, 2.2470f, 2.4110f, 2.5750f, 2.8250f, 9998.0f, 2.9960f, 3.2265f, 3.4015f,
      3.5765f, 3.7515f, 3.9260f, 4.1020f, 4.2770f, 4.4505f, 4.6270f, 4.8025f, 5.0400f, 9997.0f};

  uint8_t const hwEtaAbs = std::min(kHwEtaAbsMax + 1, std::abs(hwEta));
  return (hwEta < 0) ? -kTowerEtas[hwEtaAbs] : kTowerEtas[hwEtaAbs];
}

bool l1ScoutingRun3::calol1::validHwPhi(int16_t hwPhi) { return (hwPhi >= kHwPhiMin and hwPhi <= kHwPhiMax); }

float l1ScoutingRun3::calol1::fPhi(int16_t hwPhi) {
  //
  // Based on the implementation of l1t::CaloTools::towerPhi() in
  //  L1Trigger/L1TCalorimeter/src/CaloTools.cc
  //
  float const phi = (2 * hwPhi - 1) * M_PI / kHwPhiMax;
  return (phi > M_PI) ? (phi - 2.f * M_PI) : phi;
}
