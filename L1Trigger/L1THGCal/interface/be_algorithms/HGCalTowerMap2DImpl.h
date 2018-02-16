#ifndef __L1Trigger_L1THGCal_HGCalTowerMap2DImpl_h__
#define __L1Trigger_L1THGCal_HGCalTowerMap2DImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTowerMap.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"


class HGCalTowerMap2DImpl{

 public:

  HGCalTowerMap2DImpl( const edm::ParameterSet &conf); 
  

  void buildTowerMap2D( const std::vector<edm::Ptr<l1t::HGCalTriggerCell>> & triggerCellsPtrs,
			l1t::HGCalTowerMapBxCollection & towermaps
			);


  void eventSetup(const edm::EventSetup& es) 
    {
        triggerTools_.setEventSetup(es);
    }

  
 private:

  static const int kLayersEE_ = 28;
  static const int kLayersFH_ = 12;  
  static const unsigned kLayersBH_ = 12;
  static const unsigned kLayers_ = kLayersEE_+kLayersFH_+kLayersBH_;
  std::vector<l1t::HGCalTowerMap> towerMaps_; //towerMaps for each HGC layer

  int nEtaBins_;
  int nPhiBins_;
  std::vector<double> etaBins_;
  std::vector<double> phiBins_;
  
  bool useLayerWeights_;
  std::vector<double> layerWeights_;
  HGCalTriggerTools triggerTools_;

};



#endif
