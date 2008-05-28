#ifndef FastSimulation_EventProducer_FamosProducer_H
#define FastSimulation_EventProducer_FamosProducer_H

#include "FWCore/Framework/interface/EDProducer.h"

class FamosManager;
class ParameterSet;
class Event;
class EventSetup;

namespace HepMC { 
  class GenEvent;
}

class FamosProducer : public edm::EDProducer
{

 public:

  explicit FamosProducer(edm::ParameterSet const & p);
  virtual ~FamosProducer();
  virtual void beginRun(edm::Run & run, const edm::EventSetup & es);
  virtual void endJob();
  virtual void produce(edm::Event & e, const edm::EventSetup & c);

 private:

  FamosManager * famosManager_;
  HepMC::GenEvent * evt_;
  bool simulateMuons;
  edm::InputTag theSourceLabel;
  edm::InputTag theGenParticleLabel;
  edm::InputTag theBeamSpotLabel;

};

#endif
