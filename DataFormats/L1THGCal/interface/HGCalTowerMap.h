#ifndef DataFormats_L1TCalorimeter_HGCalTowerMap_h
#define DataFormats_L1TCalorimeter_HGCalTowerMap_h

#include "DataFormats/L1THGCal/interface/HGCalTower.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"

#include <unordered_map>

namespace l1t {

  class HGCalTowerMap;
  typedef BXVector<HGCalTowerMap> HGCalTowerMapBxCollection;

  class HGCalTowerMap {

  public:

    HGCalTowerMap(): layer_(0) {}

    HGCalTowerMap(const std::vector<unsigned short>& tower_ids, const int layer);

    // ~HGCalTowerMap();

    // const l1t::HGCalTower& tower(int iX, int iY) const;

    int layer() const { return layer_; }

    // FIXME: is this the right signature shall I pass a const ref????
    const HGCalTowerMap& operator+=(HGCalTowerMap map);

    // bool addEt(short iX, short iY, float etEm, float etHad);
    bool addEt(short bin_id, float etEm, float etHad);

    unsigned nTowers() const { return towerMap_.size(); }
    const std::map<unsigned short, l1t::HGCalTower>& towers() const { return towerMap_; }


  private:

    std::map<unsigned short, l1t::HGCalTower>  towerMap_;
    unsigned layer_;

  };

}

#endif
