#include "L1Trigger/L1THGCal/interface/backend/HGCalTowerMapImpl_SA.h"

using namespace l1thgcfirmware;

void HGCalTowerMapImplSA::runAlgorithm(const std::vector<HGCalTowerMap>& inputTowerMaps_SA,
                                       std::vector<HGCalTower>& outputTowers_SA) const {
  // Need better way to initialise the output tower map
  if (inputTowerMaps_SA.empty())
    return;
  std::vector<HGCalTowerCoord> tower_ids;
  for (const auto& tower : inputTowerMaps_SA.front().towers()) {
    tower_ids.emplace_back(tower.first, tower.second.eta(), tower.second.phi());
  }
  HGCalTowerMap towerMap(tower_ids);

  for (const auto& map : inputTowerMaps_SA) {
    towerMap += map;
  }

  for (const auto& tower : towerMap.towers()) {
    if (tower.second.etEm() > 0 || tower.second.etHad() > 0) {
      outputTowers_SA.push_back(tower.second);
    }
  }
}