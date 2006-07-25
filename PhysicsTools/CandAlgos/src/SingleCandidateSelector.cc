#include "PhysicsTools/CandAlgos/src/SingleCandidateSelector.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "PhysicsTools/CandAlgos/src/cutParser.h"
#include "PhysicsTools/CandAlgos/src/candidateMethods.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

SingleCandidateSelector::SingleCandidateSelector( const edm::ParameterSet & cfg ) {
  std::string cut = cfg.getParameter<std::string>( "cut" );
  if( ! cand::parser::cutParser( cut, reco::candidateMethods(), select_ ) ) {
    throw edm::Exception( edm::errors::Configuration, 
			  "failed to parse \"" + cut + "\"" );
  }
}

bool SingleCandidateSelector::operator()( const reco::Candidate & c ) const {
  return (*select_)( c );
}
