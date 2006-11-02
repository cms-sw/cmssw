#include "PhysicsTools/HepMCCandAlgos/src/ParticleTreeDrawer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <iostream>
#include <sstream>
using namespace std;
using namespace edm;
using namespace reco;
using namespace HepMC;

ParticleTreeDrawer::ParticleTreeDrawer( const ParameterSet & cfg ) :
  src_( cfg.getParameter<InputTag>( "src" ) ) {
}

void ParticleTreeDrawer::analyze( const Event & event, const EventSetup & es ) {
  es.getData( pdt_ );

  Handle<GenParticleCandidateCollection> particles;
  event.getByLabel( src_, particles );
  for( GenParticleCandidateCollection::const_iterator p = particles->begin();
       p != particles->end(); ++ p ) {
    if ( p->mother().isNull() ) {
      cout << "mother particle: " << endl;
      deque<string> v = decay( *p );
      for( size_t i = 0; i != v.size(); ++ i )
	cout << v[ i ] << endl;
    }
  }
}

deque<string> ParticleTreeDrawer::decay( const GenParticleCandidate &  c ) const {
  int id = c.pdgId();
  unsigned int ndau = c.numberOfDaughters();
  const DefaultConfig::ParticleData * pd = pdt_->particle( id );  
  if ( pd == 0 )
    throw edm::Exception( edm::errors::InvalidReference ) 
      << "HepMC particle with id " << id << "has no particle data";
  size_t size = 0;
  deque<string> dec; 
  for( size_t i = 0; i < ndau; ++i ) {
    const GenParticleCandidate * d = 
      dynamic_cast<const GenParticleCandidate *>( & c.daughter( i ) );
    assert( d != 0 );
    deque<string> v = decay( * d );
    size_t n = v.size(); 
    if ( dec.size() < n ) 
      dec.resize( n, string( size, ' ' ) );
    for( size_t j = 0; j < n; ++ j )
      dec[ j ] += v[ j ];
    size_t vs = v[ 0 ].size();
    for( size_t j = n; j < dec.size(); ++ j )  
      dec[ j ] += string( vs, ' ' );
    size += vs;
  }
  stringstream str;
  str << pd->name() << ",";
  string name = str.str();
  if ( name.size() < size ) {
    name.resize( size, ' ' );
    size = name.size();
  }
  dec.push_front( name );
  
  return dec;
}
