// $Id: CandCombiner.cc,v 1.13 2006/09/19 08:25:26 llista Exp $
#include "PhysicsTools/CandAlgos/src/CandCombiner.h"
#include "PhysicsTools/Parser/interface/cutParser.h"
#include "PhysicsTools/Parser/interface/MethodMap.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "PhysicsTools/Parser/interface/SingleObjectSelector.h"
using namespace reco;
using namespace edm;
using namespace std;
using namespace cand::parser;

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

