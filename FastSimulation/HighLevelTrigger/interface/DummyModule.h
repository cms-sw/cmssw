#ifndef FastSimulation_HighLevelTrigger_DummyModule_H
#define FastSimulation_HighLevelTrigger_DummyModule_H

//  The CaloRecHits copy for HLT


#include "FWCore/Framework/interface/EDProducer.h"

class ParameterSet;
class Event;
class EventSetup;

class DummyModule : public edm::EDProducer
{

 public:

  explicit DummyModule(edm::ParameterSet const & p);
  virtual ~DummyModule();
  virtual void beginJob() {;}
  virtual void endJob() {;}
  virtual void produce(edm::Event & e, const edm::EventSetup & c);

};

#endif
