#include "DQM/TrackerCommon/interface/SimpleEventFilter.h" 
#include <iostream>
 
using namespace std;
//
// -- Constructor
//
SimpleEventFilter::SimpleEventFilter( const edm::ParameterSet & pset ) {
   nInterval_ = pset.getUntrackedParameter<int>( "EventsToSkip", 10 );
   verbose_   = pset.getUntrackedParameter<bool>( "DebugOn", false );
   nEvent_    = 0;
}
//
// -- Destructor
//
SimpleEventFilter::~SimpleEventFilter() {
}
 
bool SimpleEventFilter::filter( edm::Event &, edm::EventSetup const& ) {
  nEvent_++;
  bool ret = true;
  if (nEvent_ % nInterval_ != 0) ret = false; 
  if ( verbose_ && !ret) cout << ">>> filtering event" << nEvent_ << endl; 
  return ret;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SimpleEventFilter);

 
