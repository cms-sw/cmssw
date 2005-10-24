// $Id: CandCombiner.cc,v 1.2 2005/10/03 10:12:11 llista Exp $
#include "PhysicsTools/CandAlgos/src/CandCombiner.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/CandUtils/interface/cutParser.h"
#include "PhysicsTools/CandUtils/interface/candidateMethods.h"
#include "FWCore/Utilities/interface/EDMException.h"
using namespace aod;
using namespace edm;
using namespace std;
using namespace candcombiner;

CandCombiner::CandCombiner( const ParameterSet & cfg ) :
  combiner_( 0 ) {
  string decay ( cfg.getParameter<std::string>( "decay" ) );
  int charge = 0;
  if( decayParser( decay, labels_ ) ) {
    for( vector<ConjInfo>::iterator label = labels_.begin();
         label != labels_.end(); ++label ) {
      if( label->mode_ == ConjInfo::kPlus ) charge += 1;
      else if ( label->mode_ ==ConjInfo::kMinus ) charge -= 1;
    }
  } else {
    throw edm::Exception( edm::errors::Configuration,
			  "failed to parse \"" + decay + "\"" );
  }

  boost::shared_ptr<aod::Selector> select;

  std::string cutString = cfg.getParameter<std::string>( "cut" );
  if( cutParser( cutString, candidateMethods(), select ) ) {
  } else {
    throw edm::Exception( edm::errors::Configuration,
			  "failed to parse \"" + cutString + "\"" );
  }

  int lists = labels_.size();
  if ( lists != 1 && lists != 2 ) {
    throw edm::Exception( edm::errors::LogicError,
			  "invalid number of collections" );
  }

  combiner_.reset( new TwoBodyCombiner( select, true, charge ) );

  produces<Candidates>();
}

CandCombiner::~CandCombiner() {
}

void CandCombiner::produce( Event& evt, const EventSetup& ) {
  int n = labels_.size();
  assert( n == 1 || n == 2 );
  std::vector<Handle<Candidates> > colls( n );
  for( int i = 0; i < n; ++i ) {
    evt.getByLabel( labels_[ i ].label_, colls[ i ] );
  }
  const Candidates * c1 = & * colls[ 0 ], * c2 = & * colls[ n - 1 ];
  evt.put( combiner_->combine( c1, c2 ) );
}
