#ifndef __L1Trigger_L1THGCal_HGCalTowerMapProcessorBase_h__
#define __L1Trigger_L1THGCal_HGCalTowerMapProcessorBase_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTowerMap.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"


class HGCalTowerMapProcessorBase {
 
 public:
  HGCalTowerMapProcessorBase(const edm::ParameterSet& conf) : 
    geometry_(nullptr),
    name_(conf.getParameter<std::string>("ProcessorName"))
    {}
    
  virtual ~HGCalTowerMapProcessorBase() {}

  const std::string& name() const { return name_; } 
 
  void setGeometry(const HGCalTriggerGeometryBase* const geom) { geometry_ = geom;}
  
  virtual void run(const edm::Handle<l1t::HGCalTriggerCellBxCollection>& coll,               
                   l1t::HGCalTowerMapBxCollection& collTowerMap,
           const edm::EventSetup& es) = 0;
 
 protected:
  const HGCalTriggerGeometryBase* geometry_;
 
 private:
  const std::string name_;

};


#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory< HGCalTowerMapProcessorBase* (const edm::ParameterSet&) > HGCalTowerMapFactory;

#endif
