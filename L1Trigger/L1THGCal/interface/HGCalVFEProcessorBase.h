#ifndef __L1Trigger_L1THGCal_HGCalVFEProcessorBase_h__
#define __L1Trigger_L1THGCal_HGCalVFEProcessorBase_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"

class HGCalVFEProcessorBase { 
  
 public:
  HGCalVFEProcessorBase(const edm::ParameterSet& conf) : 
    geometry_(nullptr),
    name_(conf.getParameter<std::string>("VFEProcessorName"))
    {}
  virtual ~HGCalVFEProcessorBase() {}


  virtual void vfeProcessing(const HGCEEDigiCollection&,
                             const HGCHEDigiCollection&,
                             const HGCBHDigiCollection&, 
			     const edm::EventSetup& es) = 0;
  
  const std::string& name() const { return name_; } 
  
  void setGeometry(const HGCalTriggerGeometryBase* const geom) { geometry_ = geom;}
 
  virtual void setProduces(edm::stream::EDProducer<>& prod) const = 0;
  virtual void putInEvent(edm::Event& evt) = 0;

  virtual void reset() = 0;
  
 protected:
  const HGCalTriggerGeometryBase* geometry_; 
  
 private:
  const std::string name_;

};


  
#include "FWCore/PluginManager/interface/PluginFactory.h"
typedef edmplugin::PluginFactory< HGCalVFEProcessorBase* (const edm::ParameterSet&) > HGCalVFEProcessorBaseFactory;

#endif
