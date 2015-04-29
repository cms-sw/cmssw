#include "GeneratorInterface/Core/interface/PartonShowerBsHepMCFilter.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <iostream>


using namespace edm;
using namespace std;


//constructor
PartonShowerBsHepMCFilter::PartonShowerBsHepMCFilter(const edm::ParameterSet& iConfig) :

  // particle id of the gen particles that you want to filter
  particle_id(iConfig.getParameter<int>("Particle_id")),
  // status id of the particles that you want to exclude from the filter
  exclude_status_id(iConfig.getUntrackedParameter<int>("Exclude_status_id",-1)),
  // status id of the particles that you want to filetr on
  status_id(iConfig.getUntrackedParameter<int>("Status_id",-1))

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

  if( exclude_status_id > 0. && status_id > 0.){
    std::cout << "ERROR: Skipping event: Configuration has both exclude and status id set to a value > 0. They can not be used simultaneously." << std::endl;
    return false; // skip event
  }  
  
  for ( HepMC::GenEvent::particle_const_iterator p = evt->particles_begin();
	p != evt->particles_end(); ++p ) {
	
    if( abs((*p)->pdg_id()) == particle_id ){	
      if( exclude_status_id > 0. && (*p)->status() != exclude_status_id ) 
        return true; // keep event
      else if( status_id > 0. && (*p)->status() == status_id )
        return true; // keep event
      else 
        return true; // keep event
    }

  }

  return false; // skip event

}
