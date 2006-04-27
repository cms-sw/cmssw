//#include "PluginManager/PluginManager.h"
//#include "PluginManager/ModuleDef.h"

//#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/src/VectorInputSourceFactory.h"

#include "DataFormats/Common/interface/ModuleDescription.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "FastSimulation/PileUpProducer/interface/PUProducer.h"
#include "FastSimulation/PileUpProducer/interface/PUSource.h"
#include "FastSimulation/Event/interface/FSimEvent.h"

#include "CLHEP/HepMC/GenEvent.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandPoissonQ.h"


#include <iostream>
#include <memory>

PUProducer::PUProducer(FSimEvent* aSimEvent, edm::ParameterSet const & p) :
  input(edm::VectorInputSourceFactory::get()->makeVectorInputSource(
				       p.getParameter<edm::ParameterSet>("input"), 
				       edm::InputSourceDescription()).release()),
  averageNumber_(p.getParameter<double>("averageNumber")),
  //  seed_(p.getParameter<int>("seed")),
  //  eng_(seed_),
  //  poissonDistribution_(eng_, averageNumber_),
  //  flatDistribution_(eng_,0.,1E8),
  md_(),
  mySimEvent(aSimEvent)
{
}

PUProducer::~PUProducer() {;}

void PUProducer::produce()
{

  using namespace edm; 
  
  // Get N events randomly from files
  EventPrincipalVector result;
  Handle<HepMCProduct> evt;
  //  int evts = poissonDistribution_.fire();
  int evts = RandPoissonQ::shoot(averageNumber_);

  for ( int ievt=0; ievt<evts; ++ievt ) { 

    // Select a minbias event
    //    int entry = (int) (flatDistribution_.fire());
    int entry = (int) (RandFlat::shoot() * 1E8);
    input->readMany(entry,result); // Warning! we read here only one entry !
    Event e(**(result.begin()),md_);
    e.getByType(evt);

    // Add particles to the event
    mySimEvent->addParticles(*(evt->GetEvent()));

  }
  
}
