// $Id: GenParticleCandidateProducer.cc,v 1.7 2006/11/03 11:11:49 llista Exp $
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

static const int protonId = 2212;

GenParticleCandidateProducer::GenParticleCandidateProducer( const ParameterSet & p ) :
  src_( p.getParameter<string>( "src" ) ),
  stableOnly_( p.getParameter<bool>( "stableOnly" ) ),
  excludeList_( p.getParameter<vstring>( "excludeList" ) ),
  ptMinNeutral_( p.getParameter<double>( "ptMinNeutral" ) ),
  ptMinCharged_( p.getParameter<double>( "ptMinCharged" ) ),
  keepInitialParticles_( p.getParameter<bool>( "keepInitialParticles" ) ),
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
    
    bool skip = false;
    bool pass = false;
    if (  keepInitialParticles_ ) {
      bool initialProton = ( ! part->hasParents() && part->pdg_id() == protonId );
      if ( initialProton ) pass = true;
    }
    if ( ! pass ) {
      if ( stableOnly_ && ! part->status() == 1 ) skip = true;
      else if ( excludedIds_.find( abs( part->pdg_id() ) ) != excludedIds_.end() ) skip = true;
      else {
	if ( part->status() == 1 ) {
	  double ptMin = part->particleID().threeCharge() == 0 ? ptMinNeutral_ : ptMinCharged_;
	  if ( part->momentum().perp() < ptMin ) skip = true;
	}
      }
    }
    if ( ! skip ) {
      mapIdx = idx ++;
      cands->push_back( GenParticleCandidate( part ) );
      cand = & cands->back();
    }
    if ( verbose_ )
      cout << "inserting: " 
	   << pdt->particle( part->pdg_id() )->name()
	   << " -> " <<( cand != 0 ? pdt->particle( cand->pdgId() )->name() : "--" )<< " @ " 
	   << mapIdx << endl;
    ptrMap_.insert( make_pair( part, make_pair( mapIdx, cand ) ) );
  }
  for( PtrMap::const_iterator i = ptrMap_.begin(); i != ptrMap_.end(); ++ i ) {
    int dauIdx = i->second.first;
    if ( dauIdx >= 0 ) {
      const GenParticle * part = i->first;
      GenParticleCandidate * cand = i->second.second;
      if ( verbose_ )
	cout << "mother of " << pdt->particle( cand->pdgId() )->name() << " @ " << dauIdx << " is: ";
      assert( cand != 0 );
      if ( part->hasParents() ) {
	const GenParticle * mother = part->mother();
	GenParticleCandidate * motherCand = 0;
	if ( verbose_ )
	  cout << pdt->particle( mother->pdg_id() )->name() << "; ";
	while ( motherCand == 0 && mother != 0 ) {
	  PtrMap::const_iterator f = ptrMap_.find( mother );
	  assert( f != ptrMap_.end() );
	  motherCand = f->second.second;
	  if ( motherCand == 0 ) {
	    mother = mother->hasParents() ? mother->mother() : 0;
	  }
	}
	if ( motherCand != 0 ) {
	  motherCand->addDaughter( CandidateBaseRef( GenParticleCandidateRef( ref_, dauIdx ) ) );
	  if ( verbose_ )
	    cout << pdt->particle( motherCand->pdgId() )->name() << endl;
	} else {
	  if ( verbose_ )
	    cout << "null" << endl;
	}
      } else {
	if ( verbose_ )
	  cout << "not exsisting" << endl;
      }
    }
  }

  evt.put( cands );
}


