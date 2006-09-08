#ifndef RecoAlgos_TrackSelector_h
#define RecoAlgos_TrackSelector_h
/** \class TrackSelector
 *
 * selects a subset of a track collection. Also clones
 * TrackExtra part and RecHits collection
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.9 $
 *
 * $Id: TrackSelector.h,v 1.9 2006/09/01 09:42:24 llista Exp $
 *
 */
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"

namespace helper {
  template<typename C>
  struct CollectionStoreManager;

  struct TrackCollectionStoreManager {
    TrackCollectionStoreManager() :
      selTracks_( new reco::TrackCollection ),
      selTrackExtras_( new reco::TrackExtraCollection ),
      selHits_( new TrackingRecHitCollection ) {
    }
    template<typename I>
    void cloneAndStore( const I & begin, const I & end, edm::Event & evt ) {
      using namespace reco;
      TrackingRecHitRefProd rHits = evt.template getRefBeforePut<TrackingRecHitCollection>();
      TrackExtraRefProd rTrackExtras = evt.template getRefBeforePut<TrackExtraCollection>();
      TrackRefProd rTracks = evt.template getRefBeforePut<TrackCollection>();      
      size_t idx = 0, hidx = 0;
      for( I i = begin; i != end; ++ i ) {
	const Track & trk = * * i;
	selTracks_->push_back( Track( trk ) );
	selTracks_->back().setExtra( TrackExtraRef( rTrackExtras, idx ++ ) );
	selTrackExtras_->push_back( TrackExtra( trk.outerPosition(), trk.outerMomentum(), trk.outerOk(),
						trk.innerPosition(), trk.innerMomentum(), trk.innerOk(),
						trk.outerStateCovariance(), trk.outerDetId(),
						trk.innerStateCovariance(), trk.innerDetId() ) );
	TrackExtra & tx = selTrackExtras_->back();
	for( trackingRecHit_iterator hit = trk.recHitsBegin(); hit != trk.recHitsEnd(); ++ hit ) {
	  selHits_->push_back( (*hit)->clone() );
	  tx.add( TrackingRecHitRef( rHits, hidx ++ ) );
	}
      }
    }
    void put( edm::Event & evt ) {
      evt.put( selTracks_ );
      evt.put( selTrackExtras_ );
      evt.put( selHits_ );
    }
    bool empty() const { return selTracks_->empty(); }
  private:
    std::auto_ptr<reco::TrackCollection> selTracks_;
    std::auto_ptr<reco::TrackExtraCollection> selTrackExtras_;
    std::auto_ptr<TrackingRecHitCollection> selHits_;
  };

  class TrackSelectorBase : public edm::EDFilter {
  public:
    TrackSelectorBase( const edm::ParameterSet & cfg ) {
      std::string alias( cfg.template getParameter<std::string>( "@module_label" ) );
      produces<reco::TrackCollection>().setBranchAlias( alias + "Tracks" );
      produces<reco::TrackExtraCollection>().setBranchAlias( alias + "TrackExtras" );
      produces<TrackingRecHitCollection>().setBranchAlias( alias + "RecHits" );
    }
   };

  template<>
  struct CollectionStoreManager<reco::TrackCollection> {
    typedef TrackCollectionStoreManager type;
    typedef TrackSelectorBase base;
  };
}

#endif
