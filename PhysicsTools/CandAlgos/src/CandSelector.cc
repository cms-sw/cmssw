// $Id: CandSelector.cc,v 1.10 2006/02/28 11:29:19 llista Exp $
#include <memory>
#include "PhysicsTools/CandAlgos/src/CandSelector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "PhysicsTools/CandAlgos/src/cutParser.h"
#include "PhysicsTools/CandAlgos/src/candidateMethods.h"
#include "FWCore/Utilities/interface/EDMException.h"

using namespace reco;
using namespace std;
using namespace edm;

namespace cand {
  namespace modules {
 
    CandSelector::CandSelector( const ParameterSet& cfg ) :
      CandSelectorBase( cfg.getParameter<string>( "src" ) ) {
      string cut = cfg.getParameter<string>( "cut" );
      if( ! cand::parser::cutParser( cut, candidateMethods(), select_ ) ) {
	throw Exception( errors::Configuration, 
			 "failed to parse \"" + cut + "\"" );
      }
      produces<CandidateCollection>();
    }
    
    CandSelector::~CandSelector() {
    }

  }
}

