#ifndef __L1Trigger_L1THGCal_HGCalVFEProcessorBase_h__
#define __L1Trigger_L1THGCal_HGCalVFEProcessorBase_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerSums.h"
#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"

class HGCalVFEProcessorBase { 
  
 public:
  HGCalVFEProcessorBase(const edm::ParameterSet& conf) : 
    geometry_(nullptr),
    name_(conf.getParameter<std::string>("ProcessorName"))
    {}
  virtual ~HGCalVFEProcessorBase() {}

  const std::string& name() const { return name_; } 
  
  void setGeometry(const HGCalTriggerGeometryBase* const geom) { geometry_ = geom;}

  virtual void run(const HGCEEDigiCollection&,
                   const HGCHEDigiCollection&,
                   const HGCBHDigiCollection&,
                   l1t::HGCalTriggerCellBxCollection& triggerCellColl,
                   const edm::EventSetup& es) = 0;
  
 protected:
  const HGCalTriggerGeometryBase* geometry_; 
  
 private:
  const std::string name_;

};


  
#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory< HGCalVFEProcessorBase* (const edm::ParameterSet&) > HGCalVFEProcessorBaseFactory;

#endif
