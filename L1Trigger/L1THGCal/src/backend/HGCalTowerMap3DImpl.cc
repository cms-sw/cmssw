///
/// \class HGCalTowerMap2DImpl
///
/// \author: Thomas Strebler
///
/// Description: first iteration of HGCal Tower Maps

#include "L1Trigger/L1THGCal/interface/backend/HGCalTowerMap3DImpl.h"

HGCalTowerMap3DImpl::HGCalTowerMap3DImpl() {}

void HGCalTowerMap3DImpl::buildTowerMap3D(const std::vector<edm::Ptr<l1t::HGCalTowerMap>>& towerMapsPtrs,
                                          l1t::HGCalTowerBxCollection& towers) {
  l1t::HGCalTowerMap towerMap;

  for (const auto& map : towerMapsPtrs) {
    if (towerMap.layer() == 0)
      towerMap = (*map);
    else
      towerMap += (*map);
  }

  for (const auto& tower : towerMap.towers()) {
    // FIXME: make this threshold configurable
    if (tower.second.pt() > 0)
      towers.push_back(0, tower.second);
  }
}
