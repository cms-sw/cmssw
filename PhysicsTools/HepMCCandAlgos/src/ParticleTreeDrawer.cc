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
  src_( cfg.getParameter<InputTag>( "src" ) ),
  printP4_( cfg.getUntrackedParameter<bool>( "printP4", false ) ),
  printPtEtaPhi_( cfg.getUntrackedParameter<bool>( "printPtEtaPhi", false ) ),
  printStatus_( cfg.getUntrackedParameter<bool>( "printStatus", false ) ) {
}

void ParticleTreeDrawer::analyze( const Event & event, const EventSetup & es ) {
  es.getData( pdt_ );

  Handle<GenParticleCandidateCollection> particles;
  event.getByLabel( src_, particles );
  for( GenParticleCandidateCollection::const_iterator p = particles->begin();
       p != particles->end(); ++ p ) {
    if ( p->mother().isNull() ) {
      cout << "-- decay: --" << endl;
      printDecay( * p, "" );
    }
  }
}

void ParticleTreeDrawer::printP4( const reco::GenParticleCandidate & c ) const {
  if ( printP4_ ) cout << " (" << c.px() << ", " << c.py() << ", " << c.pz() << "; " << c.energy() << ")"; 
  if ( printPtEtaPhi_ ) cout << " [" << c.pt() << ": " << c.eta() << ", " << c.phi() << "]";
  if ( printStatus_ ) cout << "{status: " << c.status() << "}";
}

void ParticleTreeDrawer::printDecay( const reco::GenParticleCandidate & c, const std::string & pre ) const {
  int id = c.pdgId();
  unsigned int ndau = c.numberOfDaughters();
  const DefaultConfig::ParticleData * pd = pdt_->particle( id );  
  assert( pd != 0 );

  cout << pd->name(); 
  printP4( c );
  cout << endl;

  if ( ndau == 0 ) return;

  bool lastLevel = true;
  for( size_t i = 0; i < ndau; ++ i )
    if ( c.daughter( i ).numberOfDaughters() != 0 ) {
      lastLevel = false;
      break;
    }      

  if ( lastLevel ) {
    cout << pre << "+-> ";
    for( size_t i = 0; i < ndau; ++ i ) {
      const GenParticleCandidate * d = 
	dynamic_cast<const GenParticleCandidate *>( & c.daughter( i ) );
      assert( d != 0 );
      const DefaultConfig::ParticleData * pd = pdt_->particle( d->pdgId() );  
      assert( pd != 0 );
      cout << pd->name();
      printP4( * d );
      if ( i != ndau - 1 )
	cout << " ";
    }
    cout << endl;
    return;
  }

  for( size_t i = 0; i < ndau; ++i ) {
    const GenParticleCandidate * d =
      dynamic_cast<const GenParticleCandidate *>( & c.daughter( i ) );
    cout << pre << "+-> ";
    string prepre( pre );
    if ( i == ndau - 1 ) prepre += "    ";
    else prepre += "|   ";
    printDecay( * d, prepre );
  }
}
