#ifndef DataFormats_L1TCalorimeter_HGCalTowerMap_h
#define DataFormats_L1TCalorimeter_HGCalTowerMap_h

#include "DataFormats/L1THGCal/interface/HGCalTower.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

#include <unordered_map>

namespace l1t {

  class HGCalTowerMap;
  struct HGCalTowerCoord;
  typedef BXVector<HGCalTowerMap> HGCalTowerMapBxCollection;

  class HGCalTowerMap {
  public:
    HGCalTowerMap() : layer_(0) {}

    HGCalTowerMap(const std::vector<l1t::HGCalTowerCoord>& tower_ids, const int layer);

    int layer() const { return layer_; }

    const HGCalTowerMap& operator+=(const HGCalTowerMap& map);

    bool addEt(const std::unordered_map<unsigned short, float>& towerIDandShares, float etEm, float etHad);

    unsigned nTowers() const { return towerMap_.size(); }
    const std::unordered_map<unsigned short, l1t::HGCalTower>& towers() const { return towerMap_; }

  private:
    std::unordered_map<unsigned short, l1t::HGCalTower> towerMap_;
    unsigned layer_;
  };

}  // namespace l1t

#endif
