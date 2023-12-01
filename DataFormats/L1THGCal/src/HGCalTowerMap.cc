#include "DataFormats/L1THGCal/interface/HGCalTowerMap.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

using namespace l1t;

HGCalTowerMap::HGCalTowerMap(const std::vector<HGCalTowerCoord>& tower_ids, const int layer = 0) : layer_(layer) {
  for (auto tower_id : tower_ids) {
    towerMap_[tower_id.rawId] = l1t::HGCalTower(0., 0., tower_id.eta, tower_id.phi, tower_id.rawId);
  }
}

const HGCalTowerMap& HGCalTowerMap::operator+=(const HGCalTowerMap& map) {
  if (nTowers() != map.nTowers()) {
    throw edm::Exception(edm::errors::StdException, "StdException")
        << "HGCalTowerMap: Trying to add HGCalTowerMaps with different bins: " << nTowers() << " and " << map.nTowers()
        << endl;
  }

  for (const auto& tower : map.towers()) {
    auto this_tower = towerMap_.find(tower.first);
    if (this_tower == towerMap_.end()) {
      throw edm::Exception(edm::errors::StdException, "StdException")
          << "HGCalTowerMap: Trying to add HGCalTowerMaps but could not find bin: " << tower.first << endl;
    } else {
      this_tower->second += tower.second;
    }
  }
  return *this;
}

bool HGCalTowerMap::addEt(const std::unordered_map<unsigned short, float>& towerIDandShares, float etEm, float etHad) {
  for (const auto& towerIDandShare : towerIDandShares) {
    auto this_tower = towerMap_.find(towerIDandShare.first);
    if (this_tower == towerMap_.end())
      return false;
    this_tower->second.addEtEm(etEm * towerIDandShare.second);
    this_tower->second.addEtHad(etHad * towerIDandShare.second);
  }
  return true;
}
