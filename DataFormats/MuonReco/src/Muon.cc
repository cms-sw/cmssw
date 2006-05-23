#include "DataFormats/MuonReco/interface/Muon.h"
using namespace reco;

Muon::Muon( const TrackRef & t, const TrackRef & s, const TrackRef & c, defaultMomentumEstimate d ) :
  track_( t ), standAlone_( s ), combined_( c ), default_( d ) { 
}


