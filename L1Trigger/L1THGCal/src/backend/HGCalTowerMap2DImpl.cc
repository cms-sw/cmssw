///
/// \class HGCalTowerMap2DImpl
///
/// \author: Thomas Strebler
///
/// Description: first iteration of HGCal Tower Maps

#include "FWCore/Utilities/interface/EDMException.h"

#include "L1Trigger/L1THGCal/interface/backend/HGCalTowerMap2DImpl.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

HGCalTowerMap2DImpl::HGCalTowerMap2DImpl(const edm::ParameterSet& conf)
    : useLayerWeights_(conf.getParameter<bool>("useLayerWeights")),
      layerWeights_(conf.getParameter<std::vector<double>>("layerWeights")),
      towerGeometryHelper_(conf.getParameter<edm::ParameterSet>("L1TTriggerTowerConfig")) {}

std::unordered_map<int, l1t::HGCalTowerMap> HGCalTowerMap2DImpl::newTowerMaps() {
  bool isNose = towerGeometryHelper_.isNose();

  std::unordered_map<int, l1t::HGCalTowerMap> towerMaps;
  for (unsigned layer = 1; layer <= triggerTools_.lastLayer(isNose); layer++) {
    // FIXME: this is hardcoded...quite ugly
    if (!isNose && layer <= triggerTools_.lastLayerEE(isNose) && layer % 2 == 0)
      continue;

    towerMaps[layer] = l1t::HGCalTowerMap(towerGeometryHelper_.getTowerCoordinates(), layer);
  }

  return towerMaps;
}
