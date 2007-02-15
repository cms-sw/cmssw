#ifndef FastSimulation_EventProducer_FamosProducer_H
#define FastSimulation_EventProducer_FamosProducer_H

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/Handle.h"

class FamosManager;

namespace HepMC { 
  class GenEvent;
}

class FamosProducer : public edm::EDProducer
{

 public:

  explicit FamosProducer(edm::ParameterSet const & p);
  virtual ~FamosProducer();
  virtual void beginJob(const edm::EventSetup & c);
  virtual void endJob();
  virtual void produce(edm::Event & e, const edm::EventSetup & c);

 private:

  FamosManager * famosManager_;
  HepMC::GenEvent * evt_;

};

#endif
