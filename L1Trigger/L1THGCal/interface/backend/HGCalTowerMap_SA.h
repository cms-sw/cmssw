#ifndef L1Trigger_L1THGCal_HGCalTowerMap_SA_h
#define L1Trigger_L1THGCal_HGCalTowerMap_SA_h

#include "L1Trigger/L1THGCal/interface/backend/HGCalTower_SA.h"

#include <unordered_map>
#include <vector>

namespace l1thgcfirmware {

  class HGCalTowerMap {
  public:
    HGCalTowerMap() = default;
    HGCalTowerMap(const std::vector<l1thgcfirmware::HGCalTowerCoord>& tower_ids);

    ~HGCalTowerMap() = default;

    HGCalTowerMap& operator+=(const HGCalTowerMap& map);

    bool addEt(short bin_id, float etEm, float etHad);

    const std::unordered_map<unsigned short, l1thgcfirmware::HGCalTower>& towers() const { return towerMap_; }

  private:
    std::unordered_map<unsigned short, l1thgcfirmware::HGCalTower> towerMap_;
  };

}  // namespace l1thgcfirmware

#endif
