#ifndef SHALLOW_EVENTDATA_PRODUCER
#define SHALLOW_EVENTDATA_PRODUCER

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <string>

class ShallowEventDataProducer : public edm::EDProducer {
 public: 
  explicit ShallowEventDataProducer(const edm::ParameterSet&);
 private: 
  void produce( edm::Event &, const edm::EventSetup & );
	std::string trig_label_="gtDigis";
};

#endif
