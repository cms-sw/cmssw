#ifndef __L1Trigger_L1THGCal_HGCalFETriggerDigiProducer_H__
#define __L1Trigger_L1THGCal_HGCalFETriggerDigiProducer_H__

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"

#include <memory>

class HGCalFETriggerDigiProducer : public edm::EDProducer {  
 public:    
  HGCalFETriggerDigiProducer(const edm::ParameterSet&);
  ~HGCalFETriggerDigiProducer() { }
  
  virtual void beginRun(const edm::Run&, 
                        const edm::EventSetup&);
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:
  // inputs
  edm::EDGetToken inputee_, inputfh_, inputbh_;
  // algorithm containers
  std::unique_ptr<HGCalTriggerGeometryBase> triggerGeometry_;
  
};

DEFINE_FWK_MODULE(HGCalFETriggerDigiProducer);

#endif
