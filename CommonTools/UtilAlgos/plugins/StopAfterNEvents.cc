#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class StopAfterNEvents : public edm::EDFilter {
public:
  StopAfterNEvents( const edm::ParameterSet & );
  ~StopAfterNEvents();
private:
  bool filter( edm::Event &, edm::EventSetup const& ) override;
  const int nMax_;
  int n_;
  const bool verbose_;
};

#include <iostream>

using namespace std;
using namespace edm;

StopAfterNEvents::StopAfterNEvents( const ParameterSet & pset ) :
  nMax_( pset.getParameter<int>( "maxEvents" ) ), n_( 0 ),
  verbose_( pset.getUntrackedParameter<bool>( "verbose", false ) ) {
}

StopAfterNEvents::~StopAfterNEvents() {
}

bool StopAfterNEvents::filter(Event&, EventSetup const&) {
  if ( n_ < 0 ) return true;
  n_ ++ ;
  bool ret = n_ <= nMax_;
  if ( verbose_ )
    cout << ">>> filtering event" << n_ << "/" << nMax_ 
	      << "(" <<  ( ret ? "true" : "false" ) << ")" << endl;
  return ret;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( StopAfterNEvents );
