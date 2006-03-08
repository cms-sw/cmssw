// $Id: TrackCandidateProducer.cc,v 1.7 2006/02/28 11:43:18 llista Exp $
#include "PhysicsTools/HepMCCandAlgos/src/HepMCCandidateProducer.h"
#include "DataFormats/HepMCCandidate/interface/HepMCCandidate.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iostream>
using namespace edm;
using namespace reco;
using namespace std;

HepMCCandidateProducer::HepMCCandidateProducer( const ParameterSet & p ) :
  source( p.getParameter<string>( "src" ) ),
  stableOnly( p.getParameter<bool>( "stableOnly" ) ) {
  produces<CandidateCollection>();
}

HepMCCandidateProducer::~HepMCCandidateProducer() { }

void HepMCCandidateProducer::produce( Event& evt, 
				      const EventSetup& ) {
  Handle<HepMCProduct> mcp;
  evt.getByLabel( source, mcp );
  const HepMC::GenEvent& mc = mcp->getHepMCData();
  auto_ptr<CandidateCollection> cands( new CandidateCollection );
  cands->reserve( mc.particles_size() );
  for( HepMC::GenEvent::particle_const_iterator p = mc.particles_begin(); 
       p != mc.particles_end(); ++ p ) {
    if ( stableOnly && (*p)->status() == 1 )
      cands->push_back( new HepMCCandidate( * p ) );
  }

  evt.put( cands );
}

