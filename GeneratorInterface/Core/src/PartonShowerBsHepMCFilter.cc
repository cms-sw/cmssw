#include "GeneratorInterface/Core/interface/PartonShowerBsHepMCFilter.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <iostream>
#include "HepPDT/ParticleID.hh"


using namespace edm;
using namespace std;


//constructor
PartonShowerBsHepMCFilter::PartonShowerBsHepMCFilter(const edm::ParameterSet& iConfig) 
{

}


//destructor
PartonShowerBsHepMCFilter::~PartonShowerBsHepMCFilter()
{

}

//
// member functions
//

// ------------ method called to produce the data  ------------
bool PartonShowerBsHepMCFilter::filter(const HepMC::GenEvent* evt)
{
  
  // loop over gen particles
  for ( HepMC::GenEvent::particle_const_iterator p = evt->particles_begin(); p != evt->particles_end(); ++p ){
	
    // check only status 2 particles
    if( (*p)->status()==2 ){
      // if one of the status 2 particles is a B-hadron, accept the event
      HepPDT::ParticleID pid((*p)->pdg_id());
      if( pid.hasBottom() ){
        return true; // accept event
      }
    }
    
  }

  return false; // skip event

}