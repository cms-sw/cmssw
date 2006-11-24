#include "PhysicsTools/HepMCCandAlgos/src/SetGenParticleMotherReference.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "FWCore/Utilities/interface/Exception.h"
using namespace edm;
using namespace reco;

SetGenParticleMotherReference::SetGenParticleMotherReference( const ParameterSet & cfg ) :
  src_( cfg.getParameter<InputTag>( "src" ) ) {
}

void SetGenParticleMotherReference::analyze( const Event & event, const EventSetup & ) {
  Handle<CandidateCollection> particles;
  event.getByLabel( src_, particles );
  
  for( CandidateCollection::const_iterator p = particles->begin();
       p != particles->end(); ++ p ) {
    for( int i = 0; i < p->numberOfDaughters(); ++ i ) {
      const Candidate & d = p->daughter( i );
      const GenParticleCandidate * dau = 
	dynamic_cast<const GenParticleCandidate *>( & d );
      if( dau == 0 )
	throw cms::Exception( "InvalidReference" ) 
	  << "input collection contains candidates that"
	  << "are not of type GenParticleCandidate";
      dau->setMother( CandidateRef( particles, i ) );
    }
  }
}
