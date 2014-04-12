#ifndef HcalLaserEventFiltProducer2012_h
#define HcalLaserEventFiltProducer2012_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "EventFilter/HcalRawToDigi/interface/HcalLaserEventFilter2012.h"

class HcalLaserEventFiltProducer2012 : public edm::EDProducer {

 public:
  explicit HcalLaserEventFiltProducer2012(const edm::ParameterSet& iConfig);
  virtual ~HcalLaserEventFiltProducer2012(){
    delete hcalLaserEventFilter2012;
} 
  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:
  virtual void endJob() override;
  HcalLaserEventFilter2012 *hcalLaserEventFilter2012;
};


#endif
