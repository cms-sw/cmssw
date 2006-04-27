#ifndef FastSimulation_PileUpProducer_PUProducer_H
#define FastSimulation_PileUpProducer_PUProducer_H

//#include "FWCore/Framework/interface/EDProducer.h"
//#include "FWCore/Framework/interface/Event.h"
//#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/VectorInputSource.h"

namespace edm { 
  class ModuleDescription;
}

namespace HepMC { 
  class GenEvent;
}

class FSimEvent;

class PUProducer 
//: public edm::EDProducer
{

 public:

  PUProducer(FSimEvent* aSimEvent, edm::ParameterSet const & p);
  virtual ~PUProducer();
  //  virtual void beginJob(const edm::EventSetup & c);
  //  virtual void endJob();
  //  virtual void 
  void produce();

 private:

  void clear();

 private:
  typedef edm::VectorInputSource::EventPrincipalVector EventPrincipalVector;
  edm::VectorInputSource* const input;
  double averageNumber_;
  //  int seed_;
  //  TripleRand eng_;
  //  RandPoisson poissonDistribution_;
  //  RandFlat flatDistribution_;
  edm::ModuleDescription md_;
  FSimEvent* mySimEvent;

  //  std::vector<const HepMC::GenEvent*>* myPileUpEvents;

};

#endif
