#ifndef __L1Trigger_L1THGCal_HGCalTowerMap2DImpl_h__
#define __L1Trigger_L1THGCal_HGCalTowerMap2DImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTowerMap.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTowerGeometryHelper.h"


class HGCalTowerMap2DImpl{

 public:

  HGCalTowerMap2DImpl( const edm::ParameterSet &conf);

  void resetTowerMaps();

  void buildTowerMap2D(const std::vector<edm::Ptr<l1t::HGCalTriggerCell>> & triggerCellsPtrs,
                      l1t::HGCalTowerMapBxCollection & towermaps);


  void eventSetup(const edm::EventSetup& es) {
        triggerTools_.eventSetup(es);
  }

 private:

  bool useLayerWeights_;
  std::vector<double> layerWeights_;
  HGCalTriggerTools triggerTools_;
  std::unordered_map<int, l1t::HGCalTowerMap> newTowerMaps();

  HGCalTriggerTowerGeometryHelper towerGeometryHelper_;

};



#endif
