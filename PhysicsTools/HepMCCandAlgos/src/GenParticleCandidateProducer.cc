// $Id: GenParticleCandidateProducer.cc,v 1.6 2006/09/29 09:33:39 llista Exp $
#include "PhysicsTools/HepMCCandAlgos/src/GenParticleCandidateProducer.h"
//#include "PhysicsTools/HepPDTProducer/interface/PDTRecord.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <fstream>
#include <vector>
#include <map>
#include <iostream>
using namespace edm;
using namespace reco;
using namespace std;
using namespace HepMC;

GenParticleCandidateProducer::GenParticleCandidateProducer( const ParameterSet & p ) :
  source( p.getParameter<string>( "src" ) ),
   stableOnly( p.getParameter<bool>( "stableOnly" ) ),
  excludeList( p.getParameter<vstring>( "excludeList" ) ),
  verbose( p.getUntrackedParameter<bool>( "verbose" ) ) {
  produces<CandidateCollection>();
}

GenParticleCandidateProducer::~GenParticleCandidateProducer() { 
}

void GenParticleCandidateProducer::beginJob( const EventSetup & es ) {
  //  const PDTRecord & rec = es.get<PDTRecord>();
  ESHandle<DefaultConfig::ParticleDataTable> pdt;
  es.getData( pdt );
  
  if ( verbose && stableOnly )
    LogInfo ( "INFO" ) << "Excluding unstable particles";
  for( vstring::const_iterator e = excludeList.begin(); 
       e != excludeList.end(); ++ e ) {
    const DefaultConfig::ParticleData * p = pdt->particle( * e );
    if ( p == 0 ) 
      throw cms::Exception( "ConfigError", "can't find particle" )
	<< "can't find particle: " << * e;
    if ( verbose )
      LogInfo ( "INFO" ) << "Excluding particle \"" << *e << "\", id: " << p->pid();
    excludedIds.insert( p->pid() );
  }
}

void GenParticleCandidateProducer::produce( Event& evt, const EventSetup& es ) {
  ESHandle<DefaultConfig::ParticleDataTable> pdt;
  es.getData( pdt );

  Handle<HepMCProduct> mcp;
  evt.getByLabel( source, mcp );
  const GenEvent * mc = mcp->GetEvent();
  if( mc == 0 ) 
    throw edm::Exception( edm::errors::InvalidReference ) 
      << "HepMC has null pointer to GenEvent" << endl;
  auto_ptr<CandidateCollection> cands( new CandidateCollection );
  ref_ = evt.getRefBeforePut<CandidateCollection>();
  size_t size = mc->particles_size();
  cands->reserve( size );
  ptrMap_.clear();
  int idx = 0;
  for( GenEvent::particle_const_iterator p = mc->particles_begin(); 
       p != mc->particles_end(); ++ p ) {
    const GenParticle * part = * p;
    int mapIdx = -1;
    GenParticleCandidate * cand = 0;
    if ( part->status() == 1 || ! stableOnly ) {
      int id = abs( part->pdg_id() );
      if ( excludedIds.find( id ) == excludedIds.end() ) {
	cand = new GenParticleCandidate( part );
	mapIdx = idx ++;
	cands->push_back( cand );
	if ( verbose ) {
	  const DefaultConfig::ParticleData * p = pdt->particle( cand->pdgId() );
	  if ( p != 0 ) {
	    cout << "Adding candidate for particle with id: " 
		 << cand->pdgId() << " (" << p->name() << "), status: " << cand->status() << endl;
	  } else {
	    cout << "Adding candidate for particle with id: " 
		 << cand->pdgId() << ", status: " << cand->status() << endl;
	  }
	}
      }
    }
    ptrMap_.insert( make_pair( part, make_pair( mapIdx, cand ) ) );
  }
  if ( verbose ) {
    cout << "Candidates built: " << cands->size() << endl;
    cout << "Pointer map entries: " << ptrMap_.size() << endl;
    cout << "Setting daughter references" << endl;
  }
  for( PtrMap::const_iterator i = ptrMap_.begin(); i != ptrMap_.end(); ++ i ) {
    int idx = i->second.first;
    if ( idx >= 0 ) {
      const GenParticle * part = i->first;
      GenParticleCandidate * cand = i->second.second;
      assert( cand != 0 );
      if ( verbose )
	cout << "Setting daughter reference for candidate " << idx << endl;
      addDaughters( cand, part );
    }
  }

  evt.put( cands );
}

void GenParticleCandidateProducer::addDaughters( GenParticleCandidate * cand, const GenParticle * part ) const {
  vector<GenParticle*> children = part->listChildren();
  if ( verbose )
    cout << "daughters found: " << children.size() << endl;
  for( vector<GenParticle*>::const_iterator c = children.begin(); c != children.end(); ++ c ) {
    PtrMap::const_iterator f = ptrMap_.find( * c );
    if ( f != ptrMap_.end() ) {
      int dauIdx = f->second.first;
      if ( dauIdx >= 0 ) {
	if ( verbose ) cout << "daughter found with index " << dauIdx << endl;
	assert( cand != 0 );
	cand->addDaughter( CandidateRef( ref_, dauIdx ) );
      } else {
	if ( verbose ) cout << "daughter marked as skipped in pointer map. Iterating over next level" << endl;
	const GenParticle * dauPart = f->first;
	addDaughters( cand, dauPart );
      }
    } else {
      if ( verbose ) cout << "daughter not found in pointer map." << endl;
    }
  }
}


