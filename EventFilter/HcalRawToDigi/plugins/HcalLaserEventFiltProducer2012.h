#ifndef HcalLaserEventFiltProducer2012_h
#define HcalLaserEventFiltProducer2012_h

#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "EventFilter/HcalRawToDigi/interface/HcalLaserEventFilter2012.h"

class HcalLaserEventFiltProducer2012 : public edm::one::EDProducer<> {
public:
  explicit HcalLaserEventFiltProducer2012(const edm::ParameterSet& iConfig);
  ~HcalLaserEventFiltProducer2012() override { delete hcalLaserEventFilter2012; }
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  void endJob() override;
  HcalLaserEventFilter2012* hcalLaserEventFilter2012;
};

#endif
