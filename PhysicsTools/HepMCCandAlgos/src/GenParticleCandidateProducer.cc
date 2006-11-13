// $Id: GenParticleCandidateProducer.cc,v 1.9 2006/11/07 16:28:38 llista Exp $
#include "PhysicsTools/HepMCCandAlgos/src/GenParticleCandidateProducer.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <fstream>
#include <vector>
#include <map>
#include <iostream>
#include <algorithm>
using namespace edm;
using namespace reco;
using namespace std;
using namespace HepMC;

static const int protonId = 2212;

GenParticleCandidateProducer::GenParticleCandidateProducer( const ParameterSet & p ) :
  src_( p.getParameter<string>( "src" ) ),
  stableOnly_( p.getParameter<bool>( "stableOnly" ) ),
  excludeList_( p.getParameter<vstring>( "excludeList" ) ),
  ptMinNeutral_( p.getParameter<double>( "ptMinNeutral" ) ),
  ptMinCharged_( p.getParameter<double>( "ptMinCharged" ) ),
  keepInitialProtons_( p.getParameter<bool>( "keepInitialProtons" ) ),
  excludeUnfragmentedClones_( p.getParameter<bool>( "excludeUnfragmentedClones" ) ) {
  produces<CandidateCollection>();
}

GenParticleCandidateProducer::~GenParticleCandidateProducer() { 
}

void GenParticleCandidateProducer::beginJob( const EventSetup & es ) {
  ESHandle<DefaultConfig::ParticleDataTable> pdt;
  es.getData( pdt );
  
  for( vstring::const_iterator e = excludeList_.begin(); 
       e != excludeList_.end(); ++ e ) {
    const DefaultConfig::ParticleData * p = pdt->particle( * e );
    if ( p == 0 ) 
      throw cms::Exception( "ConfigError", "can't find particle" )
	<< "can't find particle: " << * e;
    excludedIds_.insert( abs( p->pid() ) );
  }
}

void GenParticleCandidateProducer::produce( Event& evt, const EventSetup& es ) {
  ESHandle<DefaultConfig::ParticleDataTable> pdt;
  es.getData( pdt );

  Handle<HepMCProduct> mcp;
  evt.getByLabel( src_, mcp );
  const GenEvent * mc = mcp->GetEvent();
  if( mc == 0 ) 
    throw edm::Exception( edm::errors::InvalidReference ) 
      << "HepMC has null pointer to GenEvent" << endl;
  const size_t size = mc->particles_size();

  // copy particle pointers
  vector<const GenParticle *> particles( size );
  copy( mc->particles_begin(), mc->particles_end(), particles.begin() );
  // fill mother indices
  vector<int> mothers( size );
  for( size_t i = 0; i < size; ++ i ) {
    const GenParticle * part = particles[ i ];
    if ( part->hasParents() ) {
      const GenParticle * mother = part->mother();
      vector<const GenParticle *>::const_iterator f = find( particles.begin(), particles.end(), mother );
      assert( f != particles.end() );
      mothers[ i ] = f - particles.begin();
    } else {
      mothers[ i ] = -1;
    }
  }
  // fill daughters indices
  vector<vector<int> > daughters( size );
  for( size_t i = 0; i < size; ++ i ) {
    int mother = mothers[ i ];
    if ( mother != -1 )
      daughters[ mother ].push_back( i );
  } 

  // fill skip vector
  vector<bool> skip( size );
  for( size_t i = 0; i < size; ++ i ) {
    const GenParticle * part = particles[ i ];
    const int pdgId = part->pdg_id();
    const int status = part->status();
    
    bool skipped = false;
    bool pass = false;
    if (  keepInitialProtons_ ) {
      bool initialProton = ( mothers[ i ] == -1 && pdgId == protonId );
      if ( initialProton ) pass = true;
    }
    if ( ! pass ) {
      if ( stableOnly_ && ! status == 1 ) skipped = true;
      else if ( excludedIds_.find( abs( pdgId ) ) != excludedIds_.end() ) skipped = true;
      else {
	if ( status == 1 ) {
	  const double ptMin = part->particleID().threeCharge() == 0 ? ptMinNeutral_ : ptMinCharged_;
	  if ( part->momentum().perp() < ptMin ) skipped = true;
	}
      }
    }
    skip[ i ] = skipped;
  }

  // reverse particle order to avoir recursive calls
  for( int i = size - 1; i >= 0; -- i ) {
    const GenParticle * part = particles[ i ];
    const int pdgId = part->pdg_id();
    const int status = part->status();
    if ( status == 2 ) {
      
      int m = mothers[ i ];
      if( m != -1 ) {
	const GenParticle * mother = particles[ m ];
	if ( excludeUnfragmentedClones_ && mother->status() == 3 && mother->pdg_id() == pdgId )
	  skip[ m ] = true;
      }
      
      if ( ! skip[ i ] ) {
	bool allDaughtersSkipped = true;
	const vector<int> & ds = daughters[ i ];
	for( vector<int>::const_iterator j = ds.begin(); j != ds.end(); ++ j ) {
	  if ( ! skip[ * j ] ) {
	    allDaughtersSkipped = false;
	    break;
	  }
	}
	if ( allDaughtersSkipped ) 
	  skip[ i ] = true;
	else {
	  for( vector<int>::const_iterator j = ds.begin(); j != ds.end(); ++ j ) {
	    skip[ * j ] = false;
	  }
	}	    
      }
    }
  }

  // fill output collection and save association
  auto_ptr<CandidateCollection> cands( new CandidateCollection );
  CandidateRefProd ref = evt.getRefBeforePut<CandidateCollection>();
  cands->reserve( size );

  vector<size_t> indices;
  vector<GenParticleCandidate *> candidates;
  for( size_t i = 0; i < size; ++ i ) {
    const GenParticle * part = particles[ i ];
    GenParticleCandidate * cand = 0;
    if ( ! skip[ i ] ) {
      GenParticleCandidate * c = new GenParticleCandidate( part );
      cand = c;
      cands->push_back( c );
      indices.push_back( i );
    }
    candidates.push_back( cand );
  }
  assert( candidates.size() == size );
  assert( cands->size() == indices.size() );
  // fill references to daughters
  for( size_t i = 0; i < cands->size(); ++ i ) {
    int m = mothers[ indices[ i ] ];
    GenParticleCandidate * mother = 0;
    while ( mother == 0 && m != -1 ) {
      if ( ( mother = candidates[ m ] ) == 0 ) {
	m = ( m != -1 ) ? mothers[ m ] : -1;
      }
    }
    if ( mother != 0 ) {
      mother->addDaughter( CandidateRef( ref, i ) );
    }
  }

  evt.put( cands );
}


