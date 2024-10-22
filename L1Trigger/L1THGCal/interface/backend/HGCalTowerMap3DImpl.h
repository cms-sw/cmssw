#ifndef __L1Trigger_L1THGCal_HGCalTowerMap3DImpl_h__
#define __L1Trigger_L1THGCal_HGCalTowerMap3DImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1THGCal/interface/HGCalTowerMap.h"
#include "DataFormats/L1THGCal/interface/HGCalTower.h"

class HGCalTowerMap3DImpl {
public:
  HGCalTowerMap3DImpl();

  void buildTowerMap3D(const std::vector<edm::Ptr<l1t::HGCalTowerMap>>& towerMaps2D,
                       l1t::HGCalTowerBxCollection& towerMap);
};

#endif
