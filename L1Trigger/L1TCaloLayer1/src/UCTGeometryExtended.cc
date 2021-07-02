#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "UCTGeometryExtended.hh"
using namespace l1tcalo;

UCTRegionIndex UCTGeometryExtended::getUCTRegionNorth(UCTRegionIndex center) {
  int eta = center.first;
  uint32_t phi = center.second;
  phi += 1;
  if (phi == MaxUCTRegionsPhi)
    phi = 0;
  else if (phi > MaxUCTRegionsPhi)
    phi = 0xDEADBEEF;
  return UCTRegionIndex(eta, phi);
}

UCTRegionIndex UCTGeometryExtended::getUCTRegionSouth(UCTRegionIndex center) {
  int eta = center.first;
  uint32_t phi = center.second;
  if (phi == 0)
    phi = MaxUCTRegionsPhi - 1;
  else if (phi < MaxUCTRegionsPhi)
    phi -= 1;
  else
    phi = 0xDEADBEEF;
  return UCTRegionIndex(eta, phi);
}

UCTRegionIndex UCTGeometryExtended::getUCTRegionEast(UCTRegionIndex center) {
  int eta = center.first;
  uint32_t phi = center.second;
  eta += 1;
  if (eta == 0)
    eta = 1;  // eta = 0 is illegal, go one above
  int etaMax = MaxUCTRegionsEta;
  if (eta > etaMax)
    eta = 0;  // beyond high Eta edge - should not be used
  return UCTRegionIndex(eta, phi);
}

UCTRegionIndex UCTGeometryExtended::getUCTRegionWest(UCTRegionIndex center) {
  int eta = center.first;
  uint32_t phi = center.second;
  eta -= 1;
  if (eta == 0)
    eta = -1;  // eta = 0 is illegal, go one below
  int etaMin = -MaxUCTRegionsEta;
  if (eta < etaMin)
    eta = 0;  // beyond high Eta edge - should not be used
  return UCTRegionIndex(eta, phi);
}

UCTRegionIndex UCTGeometryExtended::getUCTRegionNE(UCTRegionIndex center) {
  int eta = center.first;
  uint32_t phi = center.second;
  phi += 1;
  if (phi == MaxUCTRegionsPhi)
    phi = 0;
  else if (phi > MaxUCTRegionsPhi)
    phi = 0xDEADBEEF;
  eta += 1;
  if (eta == 0)
    eta = 1;  // eta = 0 is illegal, go one above
  int etaMax = MaxUCTRegionsEta;
  if (eta > etaMax)
    eta = 0;  // beyond high Eta edge - should not be used
  return UCTRegionIndex(eta, phi);
}

UCTRegionIndex UCTGeometryExtended::getUCTRegionNW(UCTRegionIndex center) {
  int eta = center.first;
  uint32_t phi = center.second;
  phi += 1;
  if (phi == MaxUCTRegionsPhi)
    phi = 0;
  else if (phi > MaxUCTRegionsPhi)
    phi = 0xDEADBEEF;
  eta -= 1;
  if (eta == 0)
    eta = -1;  // eta = 0 is illegal, go one below
  int etaMin = -MaxUCTRegionsEta;
  if (eta < etaMin)
    eta = 0;  // beyond high Eta edge - should not be used
  return UCTRegionIndex(eta, phi);
}

UCTRegionIndex UCTGeometryExtended::getUCTRegionSE(UCTRegionIndex center) {
  int eta = center.first;
  uint32_t phi = center.second;
  if (phi == 0)
    phi = MaxUCTRegionsPhi - 1;
  else if (phi < MaxUCTRegionsPhi)
    phi -= 1;
  else
    phi = 0xDEADBEEF;
  eta += 1;
  if (eta == 0)
    eta = 1;  // eta = 0 is illegal, go one above
  int etaMax = MaxUCTRegionsEta;
  if (eta > etaMax)
    eta = 0;  // beyond high Eta edge - should not be used
  return UCTRegionIndex(eta, phi);
}

UCTRegionIndex UCTGeometryExtended::getUCTRegionSW(UCTRegionIndex center) {
  int eta = center.first;
  uint32_t phi = center.second;
  if (phi == 0)
    phi = MaxUCTRegionsPhi - 1;
  else if (phi < MaxUCTRegionsPhi)
    phi -= 1;
  else
    phi = 0xDEADBEEF;
  eta -= 1;
  if (eta == 0)
    eta = -1;  // eta = 0 is illegal, go one below
  int etaMin = -MaxUCTRegionsEta;
  if (eta < etaMin)
    eta = 0;  // beyond high Eta edge - should not be used
  return UCTRegionIndex(eta, phi);
}

bool UCTGeometryExtended::areNeighbors(UCTTowerIndex a, UCTTowerIndex b) {
  int diffEta = std::abs(a.first - b.first);                // Note eta values run -28 to +28, 0 excluded
  int diffPhi = std::abs(((int)a.second - (int)b.second));  // Note phi values run 1-72, with 72 & 1 as neighbors
  if ((diffEta <= 1 || (a.first == -1 && b.first == 1) || (a.first == 1 && b.first == -1)) &&
      (diffPhi <= 1 || diffPhi == 71))
    return true;
  return false;
}

bool UCTGeometryExtended::isEdgeTower(UCTTowerIndex a) {
  int eta = a.first;
  int etaMin = -MaxUCTRegionsEta;
  int etaMax = MaxUCTRegionsEta;
  if (eta == etaMin || eta == etaMax)
    return true;
  return false;
}
