#ifndef __L1Trigger_L1THGCal_HGCalTowerMapImplSA_h__
#define __L1Trigger_L1THGCal_HGCalTowerMapImplSA_h__

#include "L1Trigger/L1THGCal/interface/backend/HGCalTower_SA.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalTowerMap_SA.h"

class HGCalTowerMapImplSA {
public:
  HGCalTowerMapImplSA() = default;
  ~HGCalTowerMapImplSA() = default;

  void runAlgorithm(const std::vector<l1thgcfirmware::HGCalTowerMap>& inputTowerMaps_SA,
                    std::vector<l1thgcfirmware::HGCalTower>& outputTowers_SA) const;
};

#endif
