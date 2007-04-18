//#include "FWCore/PluginManager/interface/PluginManager.h"
//#include "FWCore/PluginManager/interface/ModuleDef.h"

//#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/src/VectorInputSourceFactory.h"

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Common/interface/Handle.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "FastSimulation/PileUpProducer/interface/PUProducer.h"
//#include "FastSimulation/PileUpProducer/plugins/PUSource.h"
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Utilities/interface/RandomEngine.h"

#include "HepMC/GenEvent.h"

#include <iostream>
#include <memory>

PUProducer::PUProducer(FSimEvent* aSimEvent, 
		       edm::ParameterSet const & p,
		       const RandomEngine* engine) :
  input(edm::VectorInputSourceFactory::get()->makeVectorInputSource(
			 p.getParameter<edm::ParameterSet>("input"), 
		         edm::InputSourceDescription()).release()),
  averageNumber_(p.getParameter<double>("averageNumber")),
  //  seed_(p.getParameter<int>("seed")),
  //  eng_(seed_),
  //  poissonDistribution_(eng_, averageNumber_),
  //  flatDistribution_(eng_,0.,1E8),
  md_(),
  mySimEvent(aSimEvent),
  random(engine)
{}

PUProducer::~PUProducer() {;}

void PUProducer::produce()
{

  using namespace edm; 
  
  // Get N events randomly from files
  EventPrincipalVector result;
  Handle<HepMCProduct> evt;
  //  int evts = poissonDistribution_.fire();
  int evts = (int) random->poissonShoot(averageNumber_);

  for ( int ievt=0; ievt<evts; ++ievt ) { 

    // Select a minbias event
    //    int entry = (int) (flatDistribution_.fire());
    int entry = (int) (random->flatShoot() * 1E8);
    input->readMany(entry,result); // Warning! we read here only one entry !
    Event e(**(result.begin()),md_);
    e.getByType(evt);

    // Add particles to the event
    mySimEvent->addParticles(*(evt->GetEvent()));

  }
  
}
