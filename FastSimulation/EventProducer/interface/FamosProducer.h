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
  virtual void beginRun(edm::Run const& run, const edm::EventSetup & es) override;
  virtual void endJob() override;
  virtual void produce(edm::Event & e, const edm::EventSetup & c) override;

 private:

  FamosManager * famosManager_;
  HepMC::GenEvent * evt_;
  bool simulateMuons;
  edm::InputTag theSourceLabel;
  edm::InputTag theGenParticleLabel;
  edm::InputTag theBeamSpotLabel;

};

#endif
