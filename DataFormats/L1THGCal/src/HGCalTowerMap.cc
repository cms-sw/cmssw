#include "DataFormats/L1THGCal/interface/HGCalTowerMap.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

using namespace l1t;

HGCalTowerMap::HGCalTowerMap(const std::vector<unsigned short>& tower_ids, const int layer=0) : layer_(layer)  {
  // FIXME: these numbers should come from the geometry....
  GlobalPoint referencePointM(-167.5, -167.5, -320.755);
  GlobalPoint referencePointP(-167.5, -167.5, 320.755);
  float bin_size = 5.;

  for(auto tower_id: tower_ids) {
    l1t::HGCalTowerID towerId(tower_id);
    GlobalPoint referencePoint = towerId.zside() < 0 ? referencePointM : referencePointP;
    GlobalPoint surface_center(referencePoint.x()+towerId.iX()*bin_size,
                               referencePoint.y()+towerId.iY()*bin_size,
                               referencePoint.z());
    l1t::HGCalTower tower(0., 0., surface_center.eta(), surface_center.phi(), tower_id);
    towerMap_[tower_id] = tower;
  }
}

// HGCalTowerMap::~HGCalTowerMap() {}


const HGCalTowerMap& HGCalTowerMap::operator+=(HGCalTowerMap map){
  if (nTowers() != map.nTowers()) {
    throw edm::Exception(edm::errors::StdException, "StdException")
      << "HGCalTowerMap: Trying to add HGCalTowerMaps with different bins: " << nTowers() << " and " << map.nTowers() <<endl;
  }

  for(auto tower: map.towers()) {
    auto this_tower = towerMap_.find(tower.first);
    if(this_tower == towerMap_.end()) {
      throw edm::Exception(edm::errors::StdException, "StdException")
        << "HGCalTowerMap: Trying to add HGCalTowerMaps but cound not find bin: " << tower.first <<endl;
    } else {
      this_tower->second+=tower.second;
    }

  }
  return *this;
}

// bool HGCalTowerMap::addEt(short iX, short iY, float etEm, float etHad) {
//   return addEt(pack_tower_ID(iX, iY), etEm, etHad);
// }

bool HGCalTowerMap::addEt(short bin_id, float etEm, float etHad) {
  auto this_tower = towerMap_.find(bin_id);
  if(this_tower == towerMap_.end()) return false;
  this_tower->second.addEtEm(etEm);
  this_tower->second.addEtHad(etHad);

  return true;
}
