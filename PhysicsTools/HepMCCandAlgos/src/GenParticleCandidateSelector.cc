// $Id: GenParticleCandidateSelector.cc,v 1.8 2006/10/29 21:09:39 llista Exp $
#include "PhysicsTools/HepMCCandAlgos/src/GenParticleCandidateSelector.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <fstream>
using namespace edm;
using namespace reco;
using namespace std;

GenParticleCandidateSelector::GenParticleCandidateSelector( const ParameterSet & p ) :
  src_( p.getParameter<string>( "src" ) ),
  stableOnly_( p.getParameter<bool>( "stableOnly" ) ),
  excludeList_( p.getParameter<vstring>( "excludeList" ) ),
  verbose_( p.getUntrackedParameter<bool>( "verbose" ) ) {
  produces<CandidateCollection>();
}

GenParticleCandidateSelector::~GenParticleCandidateSelector() { 
}

void GenParticleCandidateSelector::beginJob( const EventSetup & es ) {
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
      LogInfo ( "INFO" ) << "Excluding particle " << *e << ", id: " << p->pid();
    excludedIds_.insert( abs( p->pid() ) );
  }
}

void GenParticleCandidateSelector::produce( Event& evt, const EventSetup& ) {
  Handle<GenParticleCandidateCollection> particles;
  evt.getByLabel( src_, particles );
  auto_ptr<CandidateCollection> cands( new CandidateCollection );
  cands->reserve( particles->size() );
  size_t idx = 0;
  for( GenParticleCandidateCollection::const_iterator p = particles->begin(); 
       p != particles->end(); ++ p, ++ idx ) {
    if ( ! stableOnly_ || p->status() == 1 ) {
      int id = abs( p->pdgId() );
      if ( excludedIds_.find( id ) == excludedIds_.end() ) {
	if ( verbose_ )
	  LogInfo( "INFO" ) << "Adding candidate for particle with id: " 
			    << id << ", status: " << p->status();
	CandidateBaseRef ref( GenParticleCandidateRef( particles, idx ) );
	cands->push_back( new ShallowCloneCandidate( ref ) );
      }
    }
  }

  evt.put( cands );
}

