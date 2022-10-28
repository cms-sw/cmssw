#include "L1Trigger/L1THGCal/interface/backend/HGCalTowerMap_SA.h"

#include <unordered_map>

using namespace l1thgcfirmware;

HGCalTowerMap::HGCalTowerMap(const std::vector<l1thgcfirmware::HGCalTowerCoord>& tower_ids) {
  for (const auto tower_id : tower_ids) {
    towerMap_[tower_id.rawId] = l1thgcfirmware::HGCalTower(0., 0., tower_id.eta, tower_id.phi, tower_id.rawId);
  }
}

HGCalTowerMap& HGCalTowerMap::operator+=(const HGCalTowerMap& map) {
  for (const auto& tower : map.towers()) {
    auto this_tower = towerMap_.find(tower.first);
    if (this_tower != towerMap_.end()) {
      this_tower->second += tower.second;
    }
  }

  return *this;
}

bool HGCalTowerMap::addEt(short bin_id, float etEm, float etHad) {
  auto this_tower = towerMap_.find(bin_id);
  if (this_tower == towerMap_.end())
    return false;
  this_tower->second.addEtEm(etEm);
  this_tower->second.addEtHad(etHad);
  return true;
}