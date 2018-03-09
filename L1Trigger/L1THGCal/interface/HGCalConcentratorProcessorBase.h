#ifndef __L1Trigger_L1THGCal_HGCalConcentratorProcessorBase_h__
#define __L1Trigger_L1THGCal_HGCalConcentratorProcessorBase_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"


class HGCalConcentratorProcessorBase {
 
 public:
  HGCalConcentratorProcessorBase(const edm::ParameterSet& conf) : 
    geometry_(nullptr),
    name_(conf.getParameter<std::string>("ConcProcessorName"))
    {}
    
  virtual ~HGCalConcentratorProcessorBase() {}

  const std::string& name() const { return name_; } 
 
  void setGeometry(const HGCalTriggerGeometryBase* const geom) { geometry_ = geom;}
  
  virtual void setProduces(edm::stream::EDProducer<>& prod) const = 0;
  virtual void putInEvent(edm::Event& evt) = 0;
  virtual void bestChoiceSelect(const l1t::HGCalTriggerCellBxCollection& coll) = 0;
  virtual void thresholdSelect(const l1t::HGCalTriggerCellBxCollection& coll) = 0;
  
  virtual void reset() = 0;
 
 protected:
  const HGCalTriggerGeometryBase* geometry_;
 
 private:
  const std::string name_;

};


#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory< HGCalConcentratorProcessorBase* (const edm::ParameterSet&) > HGCalConcentratorFactory;

#endif
