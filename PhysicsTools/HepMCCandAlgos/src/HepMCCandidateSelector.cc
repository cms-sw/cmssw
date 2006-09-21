#include "PhysicsTools/HepMCCandAlgos/src/HepMCCandidateSelector.h"
#include "DataFormats/HepMCCandidate/interface/HepMCCandidate.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <CLHEP/HepMC/GenParticle.h>
#include <vector>
#include <algorithm>

using namespace std;
using namespace edm;
using namespace reco;

HepMCCandidateSelector::HepMCCandidateSelector( const ParameterSet & cfg ) :
  src_( cfg.getParameter<InputTag>( "src" ) ),
  selectMother_( false ) {
  const string 
    particleType( "particleType" ),
    particleId  ( "particleId"   ),
    motherType  ( "motherType"   ),
    motherId    ( "motherId"     );
  vector<string> 
    strings = cfg.getParameterNamesForType<string>(),
    ints    = cfg.getParameterNamesForType<int   >();
  bool 
    foundParticleType = find( strings.begin(), strings.end(), particleType ) != strings.end(),
    foundParticleId   = find( ints.begin()   , ints.end()   , particleId   ) != ints.end(),
    foundMotherType   = find( strings.begin(), strings.end(), motherType   ) != strings.end(),
    foundMotherId     = find( ints.begin()   , ints.end()   , motherId     ) != ints.end();
  int found = 0, foundMother = 0;
  if ( foundParticleType ) found++;
  if ( foundParticleId   ) found++;
  if ( foundMotherType   ) foundMother++;    
  if ( foundMotherId     ) foundMother++;    
  if ( found != 1 )
    throw edm::Exception( errors::Configuration ) 
      << "Should " 
      << ( found > 1 ? "not specify more than" : "specify" ) 
      << " one of the following option: " 
      << "\"" << particleType << "\", \"" << particleId << "\".";
  if ( foundMother > 1 )
    throw edm::Exception( errors::Configuration ) 
      << "Should not specify more than one of the following option: " 
      << "\"" << motherType << "\", \"" << motherId << "\".";
  if ( foundMother > 0 ) selectMother_ = true;
  if ( foundParticleType )    particleName_ = cfg.getParameter<string>( particleType );
  else if ( foundParticleId ) particleType_ = cfg.getParameter<int   >( particleId   );
  if ( foundMotherType )      motherName_   = cfg.getParameter<string>( motherType   );
  else if ( foundMotherId )   motherType_   = cfg.getParameter<int   >( motherId     );

  produces<CandidateCollection>();
}

void HepMCCandidateSelector::beginJob( const EventSetup & es ) {
  ESHandle<DefaultConfig::ParticleDataTable> pdt;
  es.getData( pdt );
  const DefaultConfig::ParticleData * p;
  if ( ! particleName_.empty() ) {
    p = pdt->particle( particleName_ );
    if ( p == 0 ) 
      throw cms::Exception( "ConfigError" ) 
	<< "can't find with name: " << particleName_ << " in PDT";
    particleType_ = p->pid();
  }
  if ( ! motherName_.empty() ) {
    p = pdt->particle( motherName_ );
    if ( p == 0 ) 
      throw cms::Exception( "ConfigError" ) 
	<< "can't find with name: " << particleName_ << " in PDT";
    motherType_ = p->pid();
  }
  particleType_ = abs( particleType_ );
  motherType_ = abs( motherType_ );
}

void HepMCCandidateSelector::produce( Event & evt, const EventSetup & ) {
  Handle<CandidateCollection> genParticles;
  evt.getByLabel( src_, genParticles );
  auto_ptr<CandidateCollection> selParticles;
  for( CandidateCollection::const_iterator p = genParticles->begin(); p != genParticles->end(); ++ p ) {
    typedef HepMCCandidate::GenParticleRef GenParticleRef;
    GenParticleRef g = p->get<GenParticleRef>();
    bool selected = false;
    if ( abs( g->pdg_id() ) == particleType_ ) {
      if ( selectMother_  && g->hasParents() ) {
	const HepMC::GenParticle * mother = g->mother();
	if ( abs( mother->pdg_id() ) == motherType_ ) selected = true;
      }
    }
    if ( selected ) 
      selParticles->push_back( p->clone() );
  }
  evt.put( selParticles );
}
