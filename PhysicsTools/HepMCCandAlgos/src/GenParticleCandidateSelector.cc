// $Id: GenParticleCandidateSelector.cc,v 1.1 2006/11/07 12:54:02 llista Exp $
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
  Handle<CandidateCollection> particles;
  evt.getByLabel( src_, particles );
  auto_ptr<CandidateCollection> cands( new CandidateCollection );
  cands->reserve( particles->size() );
  size_t idx = 0;
  for( CandidateCollection::const_iterator p = particles->begin(); 
       p != particles->end(); ++ p, ++ idx ) {
    int status = reco::status( * p );
    if ( ! stableOnly_ || status == 1 ) {
      int id = abs( reco::pdgId( * p ) );
      if ( excludedIds_.find( id ) == excludedIds_.end() ) {
	if ( verbose_ )
	  LogInfo( "INFO" ) << "Adding candidate for particle with id: " 
			    << id << ", status: " << status;
	CandidateBaseRef ref( CandidateRef( particles, idx ) );
	cands->push_back( new ShallowCloneCandidate( ref ) );
      }
    }
  }

  evt.put( cands );
}

