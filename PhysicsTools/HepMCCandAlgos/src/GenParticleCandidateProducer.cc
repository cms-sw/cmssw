// $Id: GenParticleCandidateProducer.cc,v 1.3 2006/11/02 10:25:09 llista Exp $
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
  src_( p.getParameter<string>( "src" ) ),
  stableOnly_( p.getParameter<bool>( "stableOnly" ) ),
  excludeList_( p.getParameter<vstring>( "excludeList" ) ),
  ptMinNeutral_( p.getParameter<double>( "ptMinNeutral" ) ),
  ptMinCharged_( p.getParameter<double>( "ptMinCharged" ) ),
  verbose_( p.getUntrackedParameter<bool>( "verbose" ) ) {
  produces<GenParticleCandidateCollection>();
}

GenParticleCandidateProducer::~GenParticleCandidateProducer() { 
}

void GenParticleCandidateProducer::beginJob( const EventSetup & es ) {
  //  const PDTRecord & rec = es.get<PDTRecord>();
  ESHandle<DefaultConfig::ParticleDataTable> pdt;
  es.getData( pdt );
  
  if ( verbose_ && stableOnly_ )
    LogInfo ( "INFO" ) << "Excluding unstable particles";
  for( vstring::const_iterator e = excludeList_.begin(); 
       e != excludeList_.end(); ++ e ) {
    const DefaultConfig::ParticleData * p = pdt->particle( * e );
    if ( p == 0 ) 
      throw cms::Exception( "ConfigError", "can't find particle" )
	<< "can't find particle: " << * e;
    if ( verbose_ )
      LogInfo ( "INFO" ) << "Excluding particle \"" << *e << "\", id: " << p->pid();
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
  auto_ptr<GenParticleCandidateCollection> cands( new GenParticleCandidateCollection );
  ref_ = evt.getRefBeforePut<GenParticleCandidateCollection>();
  size_t size = mc->particles_size();
  cands->reserve( size );
  ptrMap_.clear();
  int idx = 0;
  for( GenEvent::particle_const_iterator p = mc->particles_begin(); 
       p != mc->particles_end(); ++ p ) {
    const GenParticle * part = * p;
    int mapIdx = -1;
    reco::GenParticleCandidate * cand = 0;
    if ( ! stableOnly_ || part->status() == 1 ) {
      int id = part->pdg_id();
      if ( excludedIds_.find( abs( id ) ) == excludedIds_.end() ) {
	double ptMin = part->particleID().threeCharge() == 0 ? ptMinNeutral_ : ptMinCharged_;
	if ( part->momentum().perp() > ptMin ) {
	  mapIdx = idx ++;
	  cands->push_back( GenParticleCandidate( part ) );
	  cand = & cands->back();
	  if ( verbose_ ) {
	    const DefaultConfig::ParticleData * p = pdt->particle( id );
	    if ( p == 0 )
	      throw edm::Exception( edm::errors::InvalidReference ) 
		<< "HepMC particle with id " << id << "has no particle data" << endl;
	    if ( p != 0 ) {
	      cout << "Adding candidate for particle with id: " 
		   << cand->pdgId() << " (" << p->name() << "), status: " << cand->status() << endl;
	    } else {
	      cout << "Adding candidate for particle with id: " 
		   << cand->pdgId() << ", status: " << cand->status() 
		   << ", pt = " << cand->pt() << endl;
	    }
	  }
	}
      }
    }
    ptrMap_.insert( make_pair( part, make_pair( mapIdx, cand ) ) );
  }
  if ( verbose_ ) {
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
      addDaughters( cand, part );
    }
  }

  evt.put( cands );
}

void GenParticleCandidateProducer::addDaughters( GenParticleCandidate * cand, const GenParticle * part ) const {
  vector<GenParticle*> children = part->listChildren();
  for( vector<GenParticle*>::const_iterator c = children.begin(); c != children.end(); ++ c ) {
    PtrMap::const_iterator f = ptrMap_.find( * c );
    if ( f != ptrMap_.end() ) {
      int dauIdx = f->second.first;
      if ( dauIdx >= 0 ) {
	assert( cand != 0 );
	cand->addDaughter( CandidateBaseRef( GenParticleCandidateRef( ref_, dauIdx ) ) );
      } else {
	const GenParticle * dauPart = f->first;
	addDaughters( cand, dauPart );
      }
    }
  }
}


