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
  std::unordered_map<int, l1t::HGCalTowerMap> towerMaps;
  for (unsigned layer = 1; layer <= triggerTools_.lastLayerBH(); layer++) {
    // FIXME: this is hardcoded...quite ugly
    if (layer <= triggerTools_.lastLayerEE() && layer % 2 == 0)
      continue;
    towerMaps[layer] = l1t::HGCalTowerMap(towerGeometryHelper_.getTowerCoordinates(), layer);
  }

  return towerMaps;
}

void HGCalTowerMap2DImpl::buildTowerMap2D(const std::vector<edm::Ptr<l1t::HGCalTriggerCell>>& triggerCellsPtrs,
                                          l1t::HGCalTowerMapBxCollection& towerMaps) {
  std::unordered_map<int, l1t::HGCalTowerMap> towerMapsTmp = newTowerMaps();

  for (auto tc : triggerCellsPtrs) {
    if (triggerTools_.isNose(tc->detId()))
      continue;
    unsigned layer = triggerTools_.layerWithOffset(tc->detId());
    if (towerMapsTmp.find(layer) == towerMapsTmp.end()) {
      throw cms::Exception("Out of range")
          << "HGCalTowerMap2dImpl: Found trigger cell in layer " << layer << " for which there is no tower map\n";
    }
    // FIXME: should actually sum the energy not the Et...
    double calibPt = tc->pt();
    if (useLayerWeights_)
      calibPt = layerWeights_[layer] * tc->mipPt();

    double etEm = layer <= triggerTools_.lastLayerEE() ? calibPt : 0;
    double etHad = layer > triggerTools_.lastLayerEE() ? calibPt : 0;

    towerMapsTmp[layer].addEt(
        towerGeometryHelper_.getTriggerTowerFromTriggerCell(tc->detId(), tc->eta(), tc->phi()), etEm, etHad);
  }

  /* store towerMaps in the persistent collection */
  towerMaps.resize(0, towerMapsTmp.size());
  int i = 0;
  for (auto towerMap : towerMapsTmp) {
    towerMaps.set(0, i, towerMap.second);
    i++;
  }
}
