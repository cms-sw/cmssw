// $Id: CandSelector.cc,v 1.6 2005/10/25 08:47:05 llista Exp $
#include <memory>
#include "PhysicsTools/CandAlgos/src/CandSelector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/Candidate/interface/Candidate.h"
#include "PhysicsTools/CandAlgos/src/cutParser.h"
#include "PhysicsTools/CandAlgos/src/candidateMethods.h"
#include "FWCore/Utilities/interface/EDMException.h"

using namespace aod;
typedef Candidate::collection Candidates;

candmodules::CandSelector::CandSelector( const edm::ParameterSet& cfg ) :
  CandSelectorBase( cfg.getParameter<std::string>("src") ) {
  std::string cut = cfg.getParameter<std::string>( "cut" );
  if( cutParser( cut, candidateMethods(), select_ ) ) {
  } else {
    throw edm::Exception( edm::errors::Configuration, 
			  "failed to parse \"" + cut + "\"" );
  }
  produces<Candidates>();
}

candmodules::CandSelector::~CandSelector() {
}

