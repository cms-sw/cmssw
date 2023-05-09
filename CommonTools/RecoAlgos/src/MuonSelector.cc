#include "CommonTools/RecoAlgos/interface/MuonSelector.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace reco;

namespace helper {
  MuonCollectionStoreManager::MuonCollectionStoreManager(const edm::Handle<reco::MuonCollection> &)
      : selMuons_(new reco::MuonCollection),
        selTracks_(new reco::TrackCollection),
        selTracksExtras_(new reco::TrackExtraCollection),
        selTracksHits_(new TrackingRecHitCollection),
        selGlobalMuonTracks_(new reco::TrackCollection),
        selGlobalMuonTracksExtras_(new reco::TrackExtraCollection),
        selGlobalMuonTracksHits_(new TrackingRecHitCollection),
        selStandAloneTracks_(new reco::TrackCollection),
        selStandAloneTracksExtras_(new reco::TrackExtraCollection),
        selStandAloneTracksHits_(new TrackingRecHitCollection),
        selStripClusters_(new edmNew::DetSetVector<SiStripCluster>),
        selPixelClusters_(new edmNew::DetSetVector<SiPixelCluster>),
        selPhase2OTClusters_(new edmNew::DetSetVector<Phase2TrackerCluster1D>),
        rMuons_(),
        rTracks_(),
        rTrackExtras_(),
        rHits_(),
        rGBTracks_(),
        rGBTrackExtras_(),
        rGBHits_(),
        rSATracks_(),
        rSATrackExtras_(),
        rSAHits_(),
        clusterStorer_(),
        id_(0),
        igbd_(0),
        isad_(0),
        idx_(0),
        igbdx_(0),
        isadx_(0),
        hidx_(0),
        higbdx_(0),
        hisadx_(0),
        cloneClusters_(true) {}

  //------------------------------------------------------------------
  //!  Process a single muon.
  //------------------------------------------------------------------
  void MuonCollectionStoreManager::processMuon(const Muon &mu) {
    if (this->cloneClusters() && ((mu.globalTrack().isNonnull() && !this->clusterRefsOK(*mu.globalTrack())) ||
                                  (mu.innerTrack().isNonnull() && !this->clusterRefsOK(*mu.innerTrack()))
                                  // || (mu.outerTrack(). isNonnull() && !this->clusterRefsOK(*mu.outerTrack() ))
                                  )) {  // outer track is muon only and has no strip clusters...
      // At least until CMSSW_2_1_8, global muon track reconstruction assigns wrong hits in
      // case of a track from iterative tracking. These hits are fetched from Trajectories
      // instead of from Tracks and therefore reference temporary cluster collections.
      // As a hack we skip these muons here - they can anyway not be refitted.
      edm::LogError("BadRef") << "@SUB=MuonCollectionStoreManager::processMuon"
                              << "Skip muon: One of its tracks references "
                              << "non-available clusters!";
      return;
    }

    selMuons_->push_back(Muon(mu));
    // only tracker Muon Track
    selMuons_->back().setInnerTrack(TrackRef(rTracks_, id_++));
    TrackRef trkRef = mu.track();
    if (trkRef.isNonnull()) {
      selTracks_->push_back(Track(*trkRef));

      Track &trk = selTracks_->back();

      selTracksExtras_->push_back(TrackExtra(trk.outerPosition(),
                                             trk.outerMomentum(),
                                             trk.outerOk(),
                                             trk.innerPosition(),
                                             trk.innerMomentum(),
                                             trk.innerOk(),
                                             trk.outerStateCovariance(),
                                             trk.outerDetId(),
                                             trk.innerStateCovariance(),
                                             trk.innerDetId(),
                                             trk.seedDirection()));

      TrackExtra &tx = selTracksExtras_->back();

      auto const firstHitIndex = hidx_;
      unsigned int nHitsAdded = 0;
      for (trackingRecHit_iterator hit = trk.recHitsBegin(); hit != trk.recHitsEnd(); ++hit, ++hidx_) {
        selTracksHits_->push_back((*hit)->clone());
        TrackingRecHit *newHit = &(selTracksHits_->back());
        ++nHitsAdded;
        if (cloneClusters() && newHit->isValid() && ((*hit)->geographicalId().det() == DetId::Tracker)) {
          clusterStorer_.addCluster(*selTracksHits_, hidx_);
        }
      }  // end of for loop over tracking rec hits on this track
      tx.setHits(rHits_, firstHitIndex, nHitsAdded);

      trk.setExtra(TrackExtraRef(rTrackExtras_, idx_++));

    }  // TO trkRef.isNonnull

    // global Muon Track
    selMuons_->back().setGlobalTrack(TrackRef(rGBTracks_, igbd_++));
    trkRef = mu.combinedMuon();
    if (trkRef.isNonnull()) {
      selGlobalMuonTracks_->push_back(Track(*trkRef));
      Track &trk = selGlobalMuonTracks_->back();

      selGlobalMuonTracksExtras_->push_back(TrackExtra(trk.outerPosition(),
                                                       trk.outerMomentum(),
                                                       trk.outerOk(),
                                                       trk.innerPosition(),
                                                       trk.innerMomentum(),
                                                       trk.innerOk(),
                                                       trk.outerStateCovariance(),
                                                       trk.outerDetId(),
                                                       trk.innerStateCovariance(),
                                                       trk.innerDetId(),
                                                       trk.seedDirection()));
      TrackExtra &tx = selGlobalMuonTracksExtras_->back();
      auto const firstHitIndex = higbdx_;
      unsigned int nHitsAdded = 0;
      for (trackingRecHit_iterator hit = trk.recHitsBegin(); hit != trk.recHitsEnd(); ++hit, ++higbdx_) {
        selGlobalMuonTracksHits_->push_back((*hit)->clone());
        TrackingRecHit *newHit = &(selGlobalMuonTracksHits_->back());
        ++nHitsAdded;
        if (cloneClusters() && newHit->isValid() && ((*hit)->geographicalId().det() == DetId::Tracker)) {
          clusterStorer_.addCluster(*selGlobalMuonTracksHits_, higbdx_);
        }
      }
      tx.setHits(rGBHits_, firstHitIndex, nHitsAdded);

      trk.setExtra(TrackExtraRef(rGBTrackExtras_, igbdx_++));

    }  // GB trkRef.isNonnull()

    // stand alone Muon Track
    selMuons_->back().setOuterTrack(TrackRef(rSATracks_, isad_++));
    trkRef = mu.standAloneMuon();
    if (trkRef.isNonnull()) {
      selStandAloneTracks_->push_back(Track(*trkRef));
      Track &trk = selStandAloneTracks_->back();

      selStandAloneTracksExtras_->push_back(TrackExtra(trk.outerPosition(),
                                                       trk.outerMomentum(),
                                                       trk.outerOk(),
                                                       trk.innerPosition(),
                                                       trk.innerMomentum(),
                                                       trk.innerOk(),
                                                       trk.outerStateCovariance(),
                                                       trk.outerDetId(),
                                                       trk.innerStateCovariance(),
                                                       trk.innerDetId(),
                                                       trk.seedDirection()));
      TrackExtra &tx = selStandAloneTracksExtras_->back();
      auto const firstHitIndex = hisadx_;
      unsigned int nHitsAdded = 0;
      for (trackingRecHit_iterator hit = trk.recHitsBegin(); hit != trk.recHitsEnd(); ++hit) {
        selStandAloneTracksHits_->push_back((*hit)->clone());
        ++nHitsAdded;
        hisadx_++;
      }
      tx.setHits(rSAHits_, firstHitIndex, nHitsAdded);
      trk.setExtra(TrackExtraRef(rSATrackExtras_, isadx_++));

    }  // SA trkRef.isNonnull()
  }    // end of track, and function

  //-------------------------------------------------------------------------
  //!  Check if all references to silicon strip/pixel clusters are available.
  //-------------------------------------------------------------------------
  bool MuonCollectionStoreManager::clusterRefsOK(const reco::Track &track) const {
    for (trackingRecHit_iterator hitIt = track.recHitsBegin(); hitIt != track.recHitsEnd(); ++hitIt) {
      const TrackingRecHit &hit = **hitIt;
      if (!hit.isValid() || hit.geographicalId().det() != DetId::Tracker)
        continue;

      // So we are in the tracker - now check hit types and availability of cluster refs:
      const std::type_info &hit_type = typeid(hit);
      if (hit_type == typeid(SiPixelRecHit)) {
        if (!static_cast<const SiPixelRecHit &>(hit).cluster().isAvailable())
          return false;
      } else if (hit_type == typeid(SiStripRecHit2D)) {
        if (!static_cast<const SiStripRecHit2D &>(hit).cluster().isAvailable())
          return false;
      } else if (hit_type == typeid(SiStripRecHit1D)) {
        if (!static_cast<const SiStripRecHit1D &>(hit).cluster().isAvailable())
          return false;
      } else if (hit_type == typeid(SiStripMatchedRecHit2D)) {
        const SiStripMatchedRecHit2D &mHit = static_cast<const SiStripMatchedRecHit2D &>(hit);
        if (!mHit.monoHit().cluster().isAvailable())
          return false;
        if (!mHit.stereoHit().cluster().isAvailable())
          return false;
      } else if (hit_type == typeid(ProjectedSiStripRecHit2D)) {
        const ProjectedSiStripRecHit2D &pHit = static_cast<const ProjectedSiStripRecHit2D &>(hit);
        if (!pHit.originalHit().cluster().isAvailable())
          return false;
      } else if (hit_type == typeid(Phase2TrackerRecHit1D)) {
        if (!static_cast<const Phase2TrackerRecHit1D &>(hit).cluster().isAvailable())
          return false;
      } else {
        // std::cout << "|   It is a " << hit_type.name() << " hit !?" << std::endl;
        // Do nothing. We might end up here for FastSim hits.
      }  // end 'switch' on hit type
    }

    // No tracker hit with bad cluster found, so all fine:
    return true;
  }

  //------------------------------------------------------------------
  //!  Put Muons, tracks, track extras and hits+clusters into the event.
  //------------------------------------------------------------------
  edm::OrphanHandle<reco::MuonCollection> MuonCollectionStoreManager::put(edm::Event &evt) {
    edm::OrphanHandle<reco::MuonCollection> h;
    h = evt.put(std::move(selMuons_), "SelectedMuons");
    evt.put(std::move(selTracks_), "TrackerOnly");
    evt.put(std::move(selTracksExtras_), "TrackerOnly");
    evt.put(std::move(selTracksHits_), "TrackerOnly");
    evt.put(std::move(selGlobalMuonTracks_), "GlobalMuon");
    evt.put(std::move(selGlobalMuonTracksExtras_), "GlobalMuon");
    evt.put(std::move(selGlobalMuonTracksHits_), "GlobalMuon");
    evt.put(std::move(selStandAloneTracks_), "StandAlone");
    evt.put(std::move(selStandAloneTracksExtras_), "StandAlone");
    evt.put(std::move(selStandAloneTracksHits_), "StandAlone");
    if (cloneClusters()) {
      evt.put(std::move(selStripClusters_));
      evt.put(std::move(selPixelClusters_));
      evt.put(std::move(selPhase2OTClusters_));
    }
    return h;
  }

}  // end of namespace helper
