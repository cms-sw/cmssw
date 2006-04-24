#ifndef FastSimulation_PileUpProducer_PUProducer_H
#define FastSimulation_PileUpProducer_PUProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/VectorInputSource.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class PUProducer : public edm::EDProducer
{

 public:

  explicit PUProducer(edm::ParameterSet const & p);
  virtual ~PUProducer();
  virtual void beginJob(const edm::EventSetup & c);
  virtual void endJob();
  virtual void produce(edm::Event & e, const edm::EventSetup & c);

 private:

  edm::VectorInputSource* const input;

};

#endif
