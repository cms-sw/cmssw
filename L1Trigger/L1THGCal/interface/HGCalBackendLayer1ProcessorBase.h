#ifndef __L1Trigger_L1THGCal_HGCalBackendLayer1ProcessorBase_h__
#define __L1Trigger_L1THGCal_HGCalBackendLayer1ProcessorBase_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

class HGCalBackendLayer1ProcessorBase { 
  
 public:
  HGCalBackendLayer1ProcessorBase(const edm::ParameterSet& conf) : 
    geometry_(nullptr),
    name_(conf.getParameter<std::string>("BeProcessorLayer1Name"))
    {}
  virtual ~HGCalBackendLayer1ProcessorBase() {}

  const std::string& name() const { return name_; } 
  virtual void setGeometry(const HGCalTriggerGeometryBase* const geom) {geometry_ = geom;}
  
  virtual void setProduces2D(edm::stream::EDProducer<>& prod) const = 0;
  
  virtual void putInEvent2D(edm::Event& evt) = 0;
  
  virtual void run2D(const edm::Handle<l1t::HGCalTriggerCellBxCollection>& coll, 
		     const edm::EventSetup& es,
		     edm::Event &e) = 0;
		   
  virtual void reset2D() = 0;  

 protected:
  const HGCalTriggerGeometryBase* geometry_; 
  
 private:
  const std::string name_;

};

#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory< HGCalBackendLayer1ProcessorBase* (const edm::ParameterSet&) > HGCalBackendLayer1Factory;

#endif
