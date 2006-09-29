// $Id: CandCombiner.cc,v 1.10 2006/07/26 09:21:40 llista Exp $
#include "PhysicsTools/CandAlgos/src/CandCombiner.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/Utilities/interface/cutParser.h"
#include "PhysicsTools/Utilities/interface/MethodMap.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Utilities/interface/EDMException.h"
using namespace reco;
using namespace edm;
using namespace std;
using namespace cand::parser;
using namespace cand::modules;

CandCombiner::CandCombiner( const ParameterSet & cfg ) :
  combiner2_( 0 ), combiner3_( 0 ) {
  string decay ( cfg.getParameter<std::string>( "decay" ) );
  int charge = 0;
  vector<int> dauCharge;
  if( decayParser( decay, labels_ ) ) {
    for( vector<ConjInfo>::iterator label = labels_.begin();
         label != labels_.end(); ++label ) {
      if( label->mode_ == ConjInfo::kPlus ){
	charge += 1;
	dauCharge.push_back( 1 );
      }
      else if ( label->mode_ == ConjInfo::kMinus ) {
	charge -= 1;
	dauCharge.push_back( -1 );
      }
    }
  } else {
    throw edm::Exception( edm::errors::Configuration,
			  "failed to parse \"" + decay + "\"" );
  }

  using namespace reco::parser;

  selector_ptr select;

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

  if( lists == 2 )
    combiner2_.reset( new TwoBodyCombiner( select, true, charge ) );
  else if( lists == 3 )
    combiner3_.reset( new ThreeBodyCombiner( select, true, dauCharge, charge ) );

  produces<CandidateCollection>();
}

CandCombiner::~CandCombiner() {
}

void CandCombiner::produce( Event& evt, const EventSetup& ) {
  int n = labels_.size();
  assert( n == 2 || n == 3 );
  std::vector<Handle<CandidateCollection> > colls( n );
  for( int i = 0; i < n; ++i ) {
    evt.getByLabel( labels_[ i ].tag_, colls[ i ] );
  }

  if( n == 2 ) {
    const CandidateCollection * c1 = & * colls[ 0 ], * c2 = & * colls[ 1 ];
    evt.put( combiner2_->combine( c1, c2 ) );
  } else if( n == 3 ) {
    const CandidateCollection * c1 = & * colls[ 0 ],
      * c2 = & * colls[ 1 ], * c3 = & * colls[ 2 ];
    evt.put( combiner3_->combine( c1, c2, c3 ) );
  }
}
