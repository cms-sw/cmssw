// $Id: GenParticleCandidateProducer.cc,v 1.5 2006/11/02 20:41:25 llista Exp $
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
	}
      }
    }
    ptrMap_.insert( make_pair( part, make_pair( mapIdx, cand ) ) );
  }
  for( PtrMap::const_iterator i = ptrMap_.begin(); i != ptrMap_.end(); ++ i ) {
    int idx = i->second.first;
    if ( idx >= 0 ) {
      const GenParticle * part = i->first;
      GenParticleCandidate * cand = i->second.second;
      assert( cand != 0 );
      if ( part->hasParents() ) {
	const GenParticle * mother = part->mother();
	GenParticleCandidate * motherCand = 0;
	while ( motherCand == 0 && mother != 0 ) {
	  PtrMap::const_iterator f = ptrMap_.find( mother );
	  assert( f != ptrMap_.end() );
	  motherCand = f->second.second;
	  if ( motherCand == 0 ) {
	    mother = mother->hasParents() ? mother->mother() : 0;
	  }
	}
	if ( motherCand != 0 )
	  motherCand->addDaughter( CandidateBaseRef( GenParticleCandidateRef( ref_, idx ) ) );
      }
    }
  }

  evt.put( cands );
}


