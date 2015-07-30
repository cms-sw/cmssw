#include "RecoTracker/FinalTrackSelectors/interface/TrackCollectionCloner.h"


TrackCollectionCloner::Producer::Producer(edm::Event& ievt, TrackCollectionCloner const & cloner) :
  copyExtras_(cloner.copyExtras_), copyTrajectories_(cloner.copyTrajectories_), evt(ievt)  {
  selTracks_ = std::make_unique<reco::TrackCollection>();
  if (copyExtras_) {
    selTrackExtras_ = std::make_unique<reco::TrackExtraCollection>();
    selHits_ = std::make_unique<TrackingRecHitCollection>();
  }
  if ( copyTrajectories_ ) {
    selTrajs_ = std::make_unique< std::vector<Trajectory> >();
    selTTAss_ = std::make_unique< TrajTrackAssociationCollection >(evt.getRefBeforePut< std::vector<Trajectory> >(), evt.getRefBeforePut<reco::TrackCollection>());
  }
  
}


/// process one event
void TrackCollectionCloner::Producer::operator()(Tokens const & tokens, std::vector<unsigned int> const & selected) {
  
  auto rTracks = evt.getRefBeforePut<reco::TrackCollection>();
  
  TrackingRecHitRefProd rHits;
  reco::TrackExtraRefProd rTrackExtras;
  if (copyExtras_) {
    rHits = evt.getRefBeforePut<TrackingRecHitCollection>();
    rTrackExtras = evt.getRefBeforePut<reco::TrackExtraCollection>();
  }
  
  edm::RefProd< std::vector<Trajectory> > trajRefProd;
  if ( copyTrajectories_ ) {
    trajRefProd = evt.getRefBeforePut< std::vector<Trajectory> >();
  }

  std::vector<Trajectory> dummy;

  auto const & trajIn =  copyTrajectories_ ? tokens.trajectories(evt) : dummy;
  
  auto const & tracksIn = tokens.tracks(evt);
  for (auto k : selected) {
    auto const & trk = tracksIn[k];
    selTracks_->emplace_back( trk ); // clone and store
    if (copyTrajectories_) {
      // we assume tracks and trajectories are one-to-one and the assocMap is useless 
      selTrajs_->emplace_back(trajIn[k]);
      assert(selTrajs_->back().measurements().size()==trk.recHitsSize());
      selTTAss_->insert ( edm::Ref< std::vector<Trajectory> >(trajRefProd, selTrajs_->size() - 1),
      			  reco::TrackRef(rTracks, selTracks_->size() - 1)
      			  );
    }
    
    if (!copyExtras_) continue;
    
    // TrackExtras
    selTrackExtras_->emplace_back( trk.outerPosition(), trk.outerMomentum(), trk.outerOk(),
				   trk.innerPosition(), trk.innerMomentum(), trk.innerOk(),
				   trk.outerStateCovariance(), trk.outerDetId(),
				   trk.innerStateCovariance(), trk.innerDetId(),
				   trk.seedDirection(), trk.seedRef()
				   );
    selTracks_->back().setExtra( reco::TrackExtraRef( rTrackExtras, selTrackExtras_->size() - 1) );
    auto & tx = selTrackExtras_->back();
    tx.setResiduals(trk.residuals());
    auto nh1=trk.recHitsSize();
    tx.setHits(rHits,selHits_->size(),nh1);
    // TrackingRecHits
    for( auto hit = trk.recHitsBegin(); hit != trk.recHitsEnd(); ++ hit ) {
      selHits_->push_back( (*hit)->clone() );
    }
  }

}

TrackCollectionCloner::Producer::~Producer() {
  selTracks_->shrink_to_fit();
  auto tsize = selTracks_->size();
  evt.put(std::move(selTracks_));
  if (copyExtras_) {
    selTrackExtras_->shrink_to_fit();
    assert(selTrackExtras_->size()==tsize);
    selHits_->shrink_to_fit();
    evt.put(std::move(selTrackExtras_));
    evt.put(std::move(selHits_));
  }
  if ( copyTrajectories_ ) {
    selTrajs_->shrink_to_fit();
    assert(selTrajs_->size()==tsize);
    assert(selTTAss_->size()==tsize);
    evt.put(std::move(selTrajs_));
    evt.put(std::move(selTTAss_));
    
  }
}
