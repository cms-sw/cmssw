#include "PhysicsTools/RecoQA/src/StopAfterNEvents.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iostream>
// $Id: StopAfterNEvents.cc,v 1.2 2006/04/23 16:11:08 wmtan Exp $ 

using namespace std;
using namespace edm;

StopAfterNEvents::StopAfterNEvents( const ParameterSet & pset ) :
  nMax_( pset.getParameter<int>( "maxEvents" ) ), n_( 0 ),
  verbose_( pset.getUntrackedParameter<bool>( "verbose", false ) ) {
}

StopAfterNEvents::~StopAfterNEvents() {
}


bool StopAfterNEvents::filter( Event &, EventSetup const& ) {
  if ( n_ < 0 ) return true;
  n_ ++ ;
  bool ret = n_ <= nMax_;
  if ( verbose_ )
    cout << ">>> filtering event" << n_ << "/" << nMax_ 
	      << "(" <<  ( ret ? "true" : "false" ) << ")" << endl;
  return ret;
}
