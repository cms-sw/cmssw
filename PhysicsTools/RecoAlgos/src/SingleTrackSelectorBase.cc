#include "PhysicsTools/RecoAlgos/interface/SingleTrackSelectorBase.h"
#include "DataFormats/TrackReco/interface/Track.h"

using namespace reco;
using namespace std;

SingleTrackSelectorBase::SingleTrackSelectorBase( const edm::ParameterSet & cfg ) :
  TrackSelectorBase( cfg ) { 
}

SingleTrackSelectorBase::~SingleTrackSelectorBase() { 
}

void SingleTrackSelectorBase::select( const TrackCollection & c, vector<const Track *> & v ) const {
  for( TrackCollection::const_iterator i = c.begin(); i != c.end(); ++ i )
    if ( select( * i ) ) v.push_back( & * i );
}

bool SingleTrackSelectorBase::select( const Track & ) const {
  return true;
}

