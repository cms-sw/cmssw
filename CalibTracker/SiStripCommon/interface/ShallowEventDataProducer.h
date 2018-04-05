#ifndef SHALLOW_EVENTDATA_PRODUCER
#define SHALLOW_EVENTDATA_PRODUCER

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include <string>

class ShallowEventDataProducer : public edm::EDProducer {
 public: 
  explicit ShallowEventDataProducer(const edm::ParameterSet&);
 private: 
  void produce( edm::Event &, const edm::EventSetup & ) override;
	edm::EDGetTokenT< L1GlobalTriggerReadoutRecord > trig_token_;
	edm::EDGetTokenT< LumiScalersCollection > scalerToken_; 
};

#endif
