#ifndef __L1Trigger_L1THGCal_HGCalBackendLayer1ProcessorBase_h__
#define __L1Trigger_L1THGCal_HGCalBackendLayer1ProcessorBase_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

class HGCalBackendLayer1ProcessorBase { 
  
 public:
  HGCalBackendLayer1ProcessorBase(const edm::ParameterSet& conf) : 
    geometry_(nullptr),
    name_(conf.getParameter<std::string>("ProcessorName"))
    {}
  virtual ~HGCalBackendLayer1ProcessorBase() {}

  const std::string& name() const { return name_; }
  
  void setGeometry(const HGCalTriggerGeometryBase* const geom) {geometry_ = geom;}
    
  virtual void run(const edm::Handle<l1t::HGCalTriggerCellBxCollection>& triggercells, 
                   l1t::HGCalClusterBxCollection& clusters,
           const edm::EventSetup& es) = 0;

 protected:
  const HGCalTriggerGeometryBase* geometry_; 
  
 private:
  const std::string name_;

};

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory< HGCalBackendLayer1ProcessorBase* (const edm::ParameterSet&) > HGCalBackendLayer1Factory;

#endif
