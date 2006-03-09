// $Id: HepMCCandidateProducer.cc,v 1.1 2006/03/08 10:50:07 llista Exp $
#include "PhysicsTools/HepMCCandAlgos/src/HepMCCandidateProducer.h"
#include "DataFormats/HepMCCandidate/interface/HepMCCandidate.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <CLHEP/HepPDT/DefaultConfig.hh>
#include <CLHEP/HepPDT/TableBuilder.hh>
#include <CLHEP/HepPDT/ParticleDataTableT.hh>
#include <iostream>
#include <fstream>
using namespace edm;
using namespace reco;
using namespace std;

HepMCCandidateProducer::HepMCCandidateProducer( const ParameterSet & p ) :
  source( p.getParameter<string>( "src" ) ),
  pdtFileName( p.getParameter<string>( "pdtFileName" ) ),
  stableOnly( p.getParameter<bool>( "stableOnly" ) ),
  excludeList( p.getParameter<vstring>( "excludeList" ) ),
  pdt( "PDG table" ) {
  produces<CandidateCollection>();
  ifstream pdtFile( pdtFileName.c_str() );
  if( ! pdtFile ) 
    throw cms::Exception( "FileNotFound", "can't open pdt file" )
      << "cannot open " << pdtFileName;
  { // notice: the builder has to be destroyed 
    // in order to fill the table!
    HepPDT::TableBuilder builder( pdt );
    if( ! addPDGParticles( pdtFile, builder ) ) { 
      throw cms::Exception( "ConfigError", "can't read pdt file" )
	<< "wrong format of " << pdtFileName;
    }
  }
  for( vstring::const_iterator e = excludeList.begin(); 
       e != excludeList.end(); ++ e ) {
    const DefaultConfig::ParticleData * p = pdt.particle( * e );
    if ( p == 0 ) 
      throw cms::Exception( "ConfigError", "can't find particle" )
	<< "can't find particle: " << * e;
    excludedIds.insert( p->pid() );
  }
}

HepMCCandidateProducer::~HepMCCandidateProducer() { 
}

void HepMCCandidateProducer::produce( Event& evt, 
				      const EventSetup& ) {
  Handle<HepMCProduct> mcp;
  evt.getByLabel( source, mcp );
  const HepMC::GenEvent& mc = mcp->getHepMCData();
  auto_ptr<CandidateCollection> cands( new CandidateCollection );
  cands->reserve( mc.particles_size() );
  for( HepMC::GenEvent::particle_const_iterator p = mc.particles_begin(); 
       p != mc.particles_end(); ++ p ) {
    if ( stableOnly && (*p)->status() == 1 ) {
      int id = abs( (*p)->pdg_id() );
      if ( excludedIds.find( id ) != excludedIds.end() )
	cands->push_back( new HepMCCandidate( * p ) );
    }
  }

  evt.put( cands );
}

