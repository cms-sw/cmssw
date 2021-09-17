#ifndef __L1Trigger_L1THGCal_HGCalTowerMap2DImpl_h__
#define __L1Trigger_L1THGCal_HGCalTowerMap2DImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTowerMap.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTowerGeometryHelper.h"

class HGCalTowerMap2DImpl {
public:
  HGCalTowerMap2DImpl(const edm::ParameterSet& conf);

  void resetTowerMaps();

  template <class T>
  void buildTowerMap2D(const std::vector<edm::Ptr<T>>& ptrs, l1t::HGCalTowerMapBxCollection& towerMaps) {
    std::unordered_map<int, l1t::HGCalTowerMap> towerMapsTmp = newTowerMaps();

    for (const auto& ptr : ptrs) {
      bool isNose = triggerTools_.isNose(ptr->detId());
      unsigned layer = triggerTools_.layerWithOffset(ptr->detId());

      if (towerMapsTmp.find(layer) == towerMapsTmp.end()) {
        throw cms::Exception("Out of range")
            << "HGCalTowerMap2dImpl: Found trigger sum in layer " << layer << " for which there is no tower map\n";
      }
      // FIXME: should actually sum the energy not the Et...
      double calibPt = ptr->pt();
      if (useLayerWeights_)
        calibPt = layerWeights_[layer] * ptr->mipPt();

      double etEm = layer <= triggerTools_.lastLayerEE(isNose) ? calibPt : 0;
      double etHad = layer > triggerTools_.lastLayerEE(isNose) ? calibPt : 0;

      towerMapsTmp[layer].addEt(towerGeometryHelper_.getTriggerTower(*ptr), etEm, etHad);
    }

    /* store towerMaps in the persistent collection */
    towerMaps.resize(0, towerMapsTmp.size());
    int i = 0;
    for (const auto& towerMap : towerMapsTmp) {
      towerMaps.set(0, i, towerMap.second);
      i++;
    }
  }

  void eventSetup(const edm::EventSetup& es) {
    triggerTools_.eventSetup(es);
    towerGeometryHelper_.eventSetup(es);
  }

private:
  bool useLayerWeights_;
  std::vector<double> layerWeights_;
  HGCalTriggerTools triggerTools_;
  std::unordered_map<int, l1t::HGCalTowerMap> newTowerMaps();

  HGCalTriggerTowerGeometryHelper towerGeometryHelper_;
};

#endif
