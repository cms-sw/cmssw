// $Id: CandCombiner.cc,v 1.17 2007/06/17 09:20:57 llista Exp $
#include "PhysicsTools/CandAlgos/interface/CandCombiner.h"
#include "PhysicsTools/Parser/interface/cutParser.h"
#include "PhysicsTools/Parser/interface/MethodMap.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Utilities/interface/EDMException.h"
using namespace reco;
using namespace edm;
using namespace std;
using namespace cand::parser;

namespace reco {
  namespace modules {

    CandCombinerBase::CandCombinerBase( const ParameterSet & cfg ) {
      string decay ( cfg.getParameter<std::string>( "decay" ) );
      if( decayParser( decay, labels_ ) )
	for( vector<ConjInfo>::iterator label = labels_.begin();
	     label != labels_.end(); ++label )
	  if( label->mode_ == ConjInfo::kPlus )
	    dauCharge_.push_back( 1 );
	  else if ( label->mode_ == ConjInfo::kMinus ) 
	    dauCharge_.push_back( -1 );
	  else
	    dauCharge_.push_back( 0 );
      else
	throw edm::Exception( edm::errors::Configuration,
			      "failed to parse \"" + decay + "\"" );
      
      int lists = labels_.size();
      if ( lists != 2 && lists != 3 ) {
	throw edm::Exception( edm::errors::LogicError,
			      "invalid number of collections" );
      }
      
      produces<CandidateCollection>();
    }
    
    CandCombinerBase::~CandCombinerBase() {
    }
    
  }
}
