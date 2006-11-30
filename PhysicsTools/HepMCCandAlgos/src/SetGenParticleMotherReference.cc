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
  
  for( size_t i = 0; i != particles->size(); ++ i ) {
    const Candidate & p = (*particles)[ i ];
    for( int j = 0; j < p.numberOfDaughters(); ++ j ) {
      const Candidate & d = p.daughter( j );
      const GenParticleCandidate * dau = 
	dynamic_cast<const GenParticleCandidate *>( & d );
      if( dau == 0 )
	throw cms::Exception( "InvalidReference" ) 
	  << "input collection contains candidates that "
	  << "are not of type GenParticleCandidate";
      dau->setMother( CandidateRef( particles, i ) );
    }
  }
}
