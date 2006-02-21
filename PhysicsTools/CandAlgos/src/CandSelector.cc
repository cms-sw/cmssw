// $Id: CandSelector.cc,v 1.8 2005/12/11 19:02:14 llista Exp $
#include <memory>
#include "PhysicsTools/CandAlgos/src/CandSelector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/Candidate/interface/Candidate.h"
#include "PhysicsTools/CandAlgos/src/cutParser.h"
#include "PhysicsTools/CandAlgos/src/candidateMethods.h"
#include "FWCore/Utilities/interface/EDMException.h"

using namespace reco;

candmodules::CandSelector::CandSelector( const edm::ParameterSet& cfg ) :
  CandSelectorBase( cfg.getParameter<std::string>("src") ) {
  std::string cut = cfg.getParameter<std::string>( "cut" );
  if( cutParser( cut, candidateMethods(), select_ ) ) {
  } else {
    throw edm::Exception( edm::errors::Configuration, 
			  "failed to parse \"" + cut + "\"" );
  }
  produces<CandidateCollection>();
}

candmodules::CandSelector::~CandSelector() {
}

