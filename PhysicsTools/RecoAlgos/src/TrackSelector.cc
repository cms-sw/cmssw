#include "PhysicsTools/RecoAlgos/src/TrackSelector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include <vector>
#include <utility>
using namespace reco;
using namespace std;
using namespace edm;

TrackSelector::TrackSelector( const ParameterSet & cfg ) :
  src_( cfg.getParameter<std::string>( "src" ) ) {
  produces<TrackCollection>().setBranchAlias( src_ + "Tracks" );
  produces<TrackExtraCollection>().setBranchAlias( src_ + "TrackExtras" );
  produces<TrackingRecHitCollection>().setBranchAlias( src_ + "RecHits" );
}
 
TrackSelector::~TrackSelector() {
}

bool TrackSelector::select( const Track & ) const {
  return true;
}

void TrackSelector::produce( Event& evt, const EventSetup& ) {
  Handle<TrackCollection> tracks;
  Handle<TrackExtraCollection> trackExtras;
  evt.getByLabel( src_, tracks );
  evt.getByLabel( src_, trackExtras );
  vector<pair<size_t, size_t> > ti;
  size_t idx;

  auto_ptr<TrackingRecHitCollection> selHits( new TrackingRecHitCollection );
  idx = 0;
  for( TrackCollection::const_iterator trk = tracks->begin(); trk != tracks->end(); ++ trk, ++ idx ) {
    if( select( * trk ) ) {
      ti.push_back( make_pair( idx, trk->recHitsSize() ) );
      for( trackingRecHit_iterator hit = trk->recHitsBegin(); hit != trk->recHitsEnd(); ++ hit ) {
	selHits->push_back( (*hit)->clone() );
      }
    }
  }
  edm::OrphanHandle<TrackingRecHitCollection> hHits = evt.put( selHits );

  auto_ptr<TrackExtraCollection> selTrackExtras( new TrackExtraCollection );
  idx = 0;
  for( vector<pair<size_t, size_t> >::const_iterator t = ti.begin(); t != ti.end(); ++t  ) {
    TrackExtra te( (*trackExtras)[ t->first ] );
    for( size_t i = 0; i < t->second; ++i )
      te.add( TrackingRecHitRef( hHits, idx++ ) );
    selTrackExtras->push_back( te );
  }
  edm::OrphanHandle<TrackExtraCollection> hTrackExtras = evt.put( selTrackExtras );

  auto_ptr<TrackCollection> selTracks( new TrackCollection );
  idx = 0;
  for( vector<pair<size_t, size_t> >::const_iterator t = ti.begin(); t != ti.end(); ++t  ) {
    Track tk( (*tracks)[ t->first ] );
    tk.setExtra( TrackExtraRef( hTrackExtras, idx ++ ) );
    selTracks->push_back( tk );
  }
  evt.put( selTracks );
}
