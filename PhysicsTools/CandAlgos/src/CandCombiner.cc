// $Id: CandCombiner.cc,v 1.12 2006/08/04 11:56:38 llista Exp $
#include "PhysicsTools/CandAlgos/src/CandCombiner.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/Parser/interface/cutParser.h"
#include "PhysicsTools/Parser/interface/MethodMap.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Utilities/interface/EDMException.h"
using namespace reco;
using namespace edm;
using namespace std;
using namespace cand::parser;
using namespace cand::modules;

CandCombiner::CandCombiner( const ParameterSet & cfg ) : 
  combiner_( 0 ) {
  string decay ( cfg.getParameter<std::string>( "decay" ) );
  vector<int> dauCharge;
  if( decayParser( decay, labels_ ) ) {
    for( vector<ConjInfo>::iterator label = labels_.begin();
         label != labels_.end(); ++label ) {
      if( label->mode_ == ConjInfo::kPlus ){
	dauCharge.push_back( 1 );
      }
      else if ( label->mode_ == ConjInfo::kMinus ) {
	dauCharge.push_back( -1 );
      } else {
	dauCharge.push_back( 0 );
      }
    }
  } else {
    throw edm::Exception( edm::errors::Configuration,
			  "failed to parse \"" + decay + "\"" );
  }

  using namespace reco::parser;

  SelectorPtr select;

  std::string cutString = cfg.getParameter<std::string>( "cut" );
  if( cutParser( cutString, reco::MethodMap::methods<reco::Candidate>(), select ) ) {
  } else {
    throw edm::Exception( edm::errors::Configuration,
			  "failed to parse \"" + cutString + "\"" );
  }

  int lists = labels_.size();
  if ( lists != 2 && lists != 3 ) {
    throw edm::Exception( edm::errors::LogicError,
			  "invalid number of collections" );
  }

  combiner_.reset( new NBodyCombiner( select, true, dauCharge ) );

  produces<CandidateCollection>();
}

CandCombiner::~CandCombiner() {
}

void CandCombiner::produce( Event& evt, const EventSetup& ) {
  int n = labels_.size();
  std::vector<Handle<CandidateCollection> > colls( n );
  for( int i = 0; i < n; ++i ) {
    evt.getByLabel( labels_[ i ].tag_, colls[ i ] );
  }
  
  std::vector<const CandidateCollection *> cv;
  for( std::vector<Handle<CandidateCollection> >::const_iterator c = colls.begin();
       c != colls.end(); ++ c )
    cv.push_back( & * * c );

  evt.put( combiner_->combine( cv ) );
}
