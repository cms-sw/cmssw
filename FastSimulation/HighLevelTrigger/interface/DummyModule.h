#ifndef FastSimulation_HighLevelTrigger_DummyModule_H
#define FastSimulation_HighLevelTrigger_DummyModule_H

//  The CaloRecHits copy for HLT


#include "FWCore/Framework/interface/stream/EDProducer.h"

class ParameterSet;
class Event;
class EventSetup;

class DummyModule : public edm::stream::EDProducer <>
{

 public:

  explicit DummyModule(edm::ParameterSet const & p);
  virtual ~DummyModule();
  virtual void produce(edm::Event & e, const edm::EventSetup & c);

};

#endif
