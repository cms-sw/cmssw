#include "RecoTracker/FinalTrackSelectors/interface/TrackCollectionCloner.h"


TrackCollectionCloner::Producer::Producer(edm::Event& ievt, TrackCollectionCloner const & cloner) :
  copyExtras_(cloner.copyExtras_), copyTrajectories_(cloner.copyTrajectories_), evt(ievt)  {
  selTracks_ = std::unique_ptr<reco::TrackCollection>(new reco::TrackCollection());
  if (copyExtras_) {
    selTrackExtras_ = std::unique_ptr<reco::TrackExtraCollection>(new reco::TrackExtraCollection());
    selHits_ = std::unique_ptr<TrackingRecHitCollection>(new TrackingRecHitCollection());
  }
  if ( copyTrajectories_ ) {
    selTrajs_ = std::auto_ptr< std::vector<Trajectory> >(new std::vector<Trajectory>());
    selTTAss_ = std::auto_ptr< TrajTrackAssociationCollection >(new TrajTrackAssociationCollection());
  }
  
}


/// process one event
void TrackCollectionCloner::Producer::operator()(Tokens const & tokens, std::vector<unsigned int> const & selected) {
  edm::Handle<reco::TrackCollection> hSrcTrack;
  evt.getByToken( tokens.hSrcTrackToken_, hSrcTrack );
  
  auto rTracks = evt.template getRefBeforePut<reco::TrackCollection>();
  
  TrackingRecHitRefProd rHits;
  reco::TrackExtraRefProd rTrackExtras;
  if (copyExtras_) {
    rHits = evt.template getRefBeforePut<TrackingRecHitCollection>();
    rTrackExtras = evt.template getRefBeforePut<reco::TrackExtraCollection>();
  }
  
  typedef reco::TrackRef::key_type TrackRefKey;
  std::map<TrackRefKey, reco::TrackRef  > goodTracks;
  
  auto const & tracksIn = *hSrcTrack;
  for (auto k : selected) {
    auto const & trk = tracksIn[k];
    selTracks_->push_back( reco::Track( trk ) ); // clone and store
    if (copyTrajectories_) {
      goodTracks[k] = reco::TrackRef(rTracks, selTracks_->size() - 1);
    }

    if (!copyExtras_) continue;
    
    // TrackExtras
    selTrackExtras_->emplace_back( trk.outerPosition(), trk.outerMomentum(), trk.outerOk(),
				   trk.innerPosition(), trk.innerMomentum(), trk.innerOk(),
				   trk.outerStateCovariance(), trk.outerDetId(),
				   trk.innerStateCovariance(), trk.innerDetId(),
				   trk.seedDirection()
				   );
    selTracks_->back().setExtra( reco::TrackExtraRef( rTrackExtras, selTrackExtras_->size() - 1) );
    auto & tx = selTrackExtras_->back();
    // TrackingRecHits
    for( auto hit = trk.recHitsBegin(); hit != trk.recHitsEnd(); ++ hit ) {
      selHits_->push_back( (*hit)->clone() );
      tx.add( TrackingRecHitRef( rHits, selHits_->size() - 1) );
    }
  }
  if ( copyTrajectories_ ) {
    edm::Handle< std::vector<Trajectory> > hTraj;
    edm::Handle< TrajTrackAssociationCollection > hTTAss;
    evt.getByToken(tokens.hTTAssToken_, hTTAss);
    evt.getByToken(tokens.hTrajToken_, hTraj);
    edm::RefProd< std::vector<Trajectory> > TrajRefProd = evt.template getRefBeforePut< std::vector<Trajectory> >();
    for (size_t i = 0, n = hTraj->size(); i < n; ++i) {
      edm::Ref< std::vector<Trajectory> > trajRef(hTraj, i);
      TrajTrackAssociationCollection::const_iterator match = hTTAss->find(trajRef);
      if (match != hTTAss->end()) {
	const edm::Ref<reco::TrackCollection> &trkRef = match->val;
	auto oldKey = trkRef.key();
	auto getref = goodTracks.find(oldKey);
	if (getref != goodTracks.end()) {
	  // do the clone
	  selTrajs_->push_back( Trajectory(*trajRef) );
	  selTTAss_->insert ( edm::Ref< std::vector<Trajectory> >(TrajRefProd, selTrajs_->size() - 1),
			      getref->second );
	}
      }
    }
  }
}

TrackCollectionCloner::Producer::~Producer() {
  selTracks_->shrink_to_fit();
  evt.put(std::move(selTracks_));
  if (copyExtras_) {
    selTrackExtras_->shrink_to_fit();
    selHits_->shrink_to_fit();
    evt.put(std::move(selTrackExtras_));
    evt.put(std::move(selHits_));
    if ( copyTrajectories_ ) {
      evt.put(std::move(selTrajs_));
      selTrajs_->shrink_to_fit();
      evt.put(std::move(selTTAss_));
    }
  }
}
