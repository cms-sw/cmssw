// $Id: HepMCCandidateProducer.cc,v 1.4 2006/03/14 17:05:14 llista Exp $
#include "PhysicsTools/HepMCCandAlgos/src/HepMCCandidateProducer.h"
//#include "PhysicsTools/HepPDTProducer/interface/PDTRecord.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "DataFormats/HepMCCandidate/interface/HepMCCandidate.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <fstream>
using namespace edm;
using namespace reco;
using namespace std;

HepMCCandidateProducer::HepMCCandidateProducer( const ParameterSet & p ) :
  source( p.getParameter<string>( "src" ) ),
   stableOnly( p.getParameter<bool>( "stableOnly" ) ),
  excludeList( p.getParameter<vstring>( "excludeList" ) ),
  verbose( p.getUntrackedParameter<bool>( "verbose" ) ) {
  produces<CandidateCollection>();
}

HepMCCandidateProducer::~HepMCCandidateProducer() { 
}

void HepMCCandidateProducer::beginJob( const EventSetup & es ) {
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
      LogInfo ( "INFO" ) << "Excluding particle " << *e << ", id: " << p->pid();
    excludedIds.insert( p->pid() );
  }
}

void HepMCCandidateProducer::produce( Event& evt, const EventSetup& ) {
  Handle<HepMCProduct> mcp;
  evt.getByLabel( source, mcp );
  const HepMC::GenEvent& mc = mcp->getHepMCData();
  auto_ptr<CandidateCollection> cands( new CandidateCollection );
  cands->reserve( mc.particles_size() );
  for( HepMC::GenEvent::particle_const_iterator p = mc.particles_begin(); 
       p != mc.particles_end(); ++ p ) {
    if ( (*p)->status() == 1 || ! stableOnly ) {
      int id = abs( (*p)->pdg_id() );
      if ( excludedIds.find( id ) == excludedIds.end() ) {
	if ( verbose )
	  LogInfo( "INFO" ) << "Adding candidate for particle with id: " 
			    << (*p)->pdg_id() << ", status: " << (*p)->status();
	cands->push_back( new HepMCCandidate( * p ) );
      }
    }
  }

  evt.put( cands );
}

