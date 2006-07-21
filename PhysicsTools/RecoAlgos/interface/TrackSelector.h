#ifndef RecoAlgos_TrackSelector_h
#define RecoAlgos_TrackSelector_h
/** \class TrackSelector
 *
 * selects a subset of a track collection. Also clones
 * TrackExtra part and RecHits collection
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.1 $
 *
 * $Id: TrackSelector.h,v 1.1 2006/07/21 10:27:05 llista Exp $
 *
 */

#include "FWCore/Framework/interface/EDFilter.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "FWCore/Framework/interface/EventPrincipal.h" 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include <utility>
#include <vector>

template<typename S>
class TrackSelector : public edm::EDFilter {
public:
  /// constructor 
  explicit TrackSelector( const edm::ParameterSet & );
  /// destructor
  virtual ~TrackSelector();
  
private:
  /// process one event
  virtual bool filter( edm::Event&, const edm::EventSetup& );
  /// source collection label
  std::string src_;
  /// filter event
  bool filter_;
  /// track collection selector
  S selector_;
};

template<typename S>
TrackSelector<S>::TrackSelector( const edm::ParameterSet & cfg ) :
  src_( cfg.template getParameter<std::string>( "src" ) ),
  filter_( cfg.template getParameter<bool>( "filter" ) ),
  selector_( cfg ){
  std::string alias( cfg.template getParameter<std::string>( "@module_label" ) );
  produces<reco::TrackCollection>().setBranchAlias( alias + "Tracks" );
  produces<reco::TrackExtraCollection>().setBranchAlias( alias + "TrackExtras" );
  produces<TrackingRecHitCollection>().setBranchAlias( alias + "RecHits" );
}
 
template<typename S>
TrackSelector<S>::~TrackSelector() {
}

template<typename S>
bool TrackSelector<S>::filter( edm::Event& evt, const edm::EventSetup& ) {
  using namespace edm;
  using namespace std;
  using namespace reco;
  Handle<TrackCollection> tracks;
  evt.getByLabel( src_, tracks );
  auto_ptr<TrackCollection> selTracks( new TrackCollection );
  auto_ptr<TrackExtraCollection> selTrackExtras( new TrackExtraCollection );
  auto_ptr<TrackingRecHitCollection> selHits( new TrackingRecHitCollection );

  TrackingRecHitRefProd rHits = evt.template getRefBeforePut<TrackingRecHitCollection>();
  TrackExtraRefProd rTrackExtras = evt.template getRefBeforePut<TrackExtraCollection>();
  TrackRefProd rTracks = evt.template getRefBeforePut<TrackCollection>();
  size_t idx = 0, hidx = 0;
  selector_.select( * tracks );
  for( typename S::const_iterator i = selector_.begin(); i != selector_.end(); ++ i ) {
    const Track & trk = * * i;
    selTracks->push_back( Track( trk ) );
    selTracks->back().setExtra( TrackExtraRef( rTrackExtras, idx ++ ) );
    selTrackExtras->push_back( TrackExtra ( trk.outerPosition(), trk.outerMomentum(), trk.outerOk() ) );
    TrackExtra & tx = selTrackExtras->back();
    for( trackingRecHit_iterator hit = trk.recHitsBegin(); hit != trk.recHitsEnd(); ++ hit ) {
      selHits->push_back( (*hit)->clone() );
      tx.add( TrackingRecHitRef( rHits, hidx ++ ) );
    }
  }
  
  if ( filter_ && selTracks->empty() ) return false;
  evt.put( selTracks );
  evt.put( selTrackExtras );
  evt.put( selHits );
  return true;
}



#endif
