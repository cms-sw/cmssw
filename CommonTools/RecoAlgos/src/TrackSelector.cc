#include "CommonTools/RecoAlgos/interface/TrackSelector.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"

#include <DataFormats/SiStripDetId/interface/SiStripDetId.h>

template<typename RecHitType, typename ClusterRefType>
void 
helper::TrackCollectionStoreManager::ClusterHitRecord<RecHitType,ClusterRefType>::
    rekey(TrackingRecHitCollection &hits, const ClusterRefType &newRef) const  
{
    //std::cout << "Rekeying hit with index " << index_ << ", detid = " << detid_ << std::endl;
    TrackingRecHit & genericHit = hits[index_]; 
    RecHitType * hit = 0;
    if (genericHit.geographicalId().rawId() == detid_) { // a hit on this det, so it's simple
        hit = static_cast<RecHitType *>(&genericHit);
    } else { // projected or matched rechit
        assert ( typeid(RecHitType) == typeid(SiStripRecHit2D) ); // must not happen for pixel
        if (typeid(genericHit) == typeid(SiStripMatchedRecHit2D)) {
            SiStripMatchedRecHit2D & mhit = static_cast<SiStripMatchedRecHit2D &>(genericHit);
            // I need the reinterpret_cast because the compiler sees this code even if RecHitType = PixelRecHit
            hit = reinterpret_cast<RecHitType *>(SiStripDetId(detid_).stereo() ? mhit.stereoHit() : mhit.monoHit());
        } else {
            assert (typeid(genericHit) == typeid(ProjectedSiStripRecHit2D)) ; // no option left, so this shoud be true
            ProjectedSiStripRecHit2D &phit = static_cast<ProjectedSiStripRecHit2D &>(genericHit);
            hit = reinterpret_cast<RecHitType *>(& phit.originalHit());
        }
    }
    assert (hit != 0);
    assert ( hit->cluster() == ref_ ); // otherwise something went wrong
    hit->setClusterRef(newRef);
}

using namespace reco;

namespace helper 
{

  TrackCollectionStoreManager::
  TrackCollectionStoreManager(const edm::Handle<reco::TrackCollection> & ) 
    :
    selTracks_( new reco::TrackCollection ),
    selTrackExtras_( new reco::TrackExtraCollection ),
    selHits_( new TrackingRecHitCollection ),
    selStripClusters_( new edmNew::DetSetVector<SiStripCluster> ),
    selPixelClusters_( new edmNew::DetSetVector<SiPixelCluster> ),
    rTracks_(), rTrackExtras_(), rHits_(), rStripClusters_(), rPixelClusters_(),
    idx_(0), hidx_(0),
    cloneClusters_ (true)
  {
  }
  
  
  
  //------------------------------------------------------------------
  //!  Process a single track.  
  //------------------------------------------------------------------
  void 
  TrackCollectionStoreManager::
  processTrack( const Track & trk ) 
  {
    //std::cout << "Process track " << idx_ << ", pt = " << trk.pt() << ", eta = " << trk.eta() << ", phi = " << trk.phi() << std::endl;
    selTracks_->push_back( Track( trk ) );
    selTracks_->back().setExtra( TrackExtraRef( rTrackExtras_, idx_ ++ ) );
    selTrackExtras_->push_back( TrackExtra( trk.outerPosition(), trk.outerMomentum(), trk.outerOk(),
					    trk.innerPosition(), trk.innerMomentum(), trk.innerOk(),
					    trk.outerStateCovariance(), trk.outerDetId(),
					    trk.innerStateCovariance(), trk.innerDetId(),
					    trk.seedDirection() ) );
    TrackExtra & tx = selTrackExtras_->back();
    for( trackingRecHit_iterator hit = trk.recHitsBegin(); hit != trk.recHitsEnd(); ++ hit ) {
        //std::cout << "\\-- Process hit " << hidx_ << ", detid = " << (*hit)->geographicalId()() << std::endl;
        selHits_->push_back( (*hit)->clone() );
        TrackingRecHit * newHit = & (selHits_->back());
        tx.add( TrackingRecHitRef( rHits_, hidx_ ++ ) );

        //--- Skip the rest for this hit if we don't want to clone the cluster.
        //--- The copy constructer in the rec hit will copy the link properly.
        //
        if (cloneClusters() == false)
            continue;       // go to the next hit on the track

        //std::cout << "|   I'm cloing clusters" << std::endl;

        const DetId detId( (*hit)->geographicalId() );
        if (newHit->isValid() && (detId.det() == DetId::Tracker)) {
            //std::cout << "|   It is a tracker hit" << std::endl;

            const std::type_info & hit_type = typeid(*newHit);
            if (hit_type == typeid(SiPixelRecHit)) {
                //std::cout << "|  It is a Pixel hit !!" << std::endl;
                pixelClusterRecords_.push_back( PixelClusterHitRecord( static_cast<SiPixelRecHit &>(*newHit), hidx_ - 1) );
            } else if (hit_type == typeid(SiStripRecHit2D)) {
                //std::cout << "|   It is a SiStripRecHit2D hit !!" << std::endl;
                stripClusterRecords_.push_back( StripClusterHitRecord( static_cast<SiStripRecHit2D &>(*newHit), hidx_ - 1) );
            } else if (hit_type == typeid(SiStripMatchedRecHit2D)) {      
                //std::cout << "|   It is a SiStripMatchedRecHit2D hit !!" << std::endl;
                SiStripMatchedRecHit2D & mhit = static_cast<SiStripMatchedRecHit2D &>(*newHit);
                stripClusterRecords_.push_back( StripClusterHitRecord( *mhit.monoHit()  , hidx_ - 1) );
                stripClusterRecords_.push_back( StripClusterHitRecord( *mhit.stereoHit(), hidx_ - 1) );
            } else if (hit_type == typeid(ProjectedSiStripRecHit2D)) {
                //std::cout << "|   It is a ProjectedSiStripRecHit2D hit !!" << std::endl;
                ProjectedSiStripRecHit2D & phit = static_cast<ProjectedSiStripRecHit2D &>(*newHit);
                stripClusterRecords_.push_back( StripClusterHitRecord( phit.originalHit(), hidx_ - 1) );
            } else {
                //std::cout << "|   It is a " << hit_type.name() << " hit !?" << std::endl;
                // do nothing. We might end up here for FastSim hits.
            } // end 'switch' on hit type
        } // end if it was a tracker hit
    } // end of for loop over tracking rec hits on this track
  } // end of track, and function

  void
  TrackCollectionStoreManager::
  processAllClusters() 
  {
      if (!pixelClusterRecords_.empty()) {
          processClusters<SiPixelRecHit,  SiPixelCluster>(pixelClusterRecords_, *selPixelClusters_, rPixelClusters_ );
      }
      if (!stripClusterRecords_.empty()) {
          processClusters<SiStripRecHit2D,SiStripCluster>(stripClusterRecords_, *selStripClusters_, rStripClusters_ ); 
      }
  }

  template<typename HitType, typename ClusterType>
  void
  TrackCollectionStoreManager::
  processClusters( std::vector<ClusterHitRecord<HitType> >      & clusterRecords,
              edmNew::DetSetVector<ClusterType>                 & dsv,
              edm::RefProd< edmNew::DetSetVector<ClusterType> > & refprod )
  {
      std::sort(clusterRecords.begin(), clusterRecords.end()); // this sorts them by detid 
      typedef typename std::vector<ClusterHitRecord<HitType> >::const_iterator RIT;
      RIT it = clusterRecords.begin(), end = clusterRecords.end();
      size_t clusters = 0;
      while (it != end) {
          RIT it2 = it;
          uint32_t detid = it->detid();

          // first isolate all clusters on the same detid
          while ( (it2 != end) && (it2->detid() == detid)) {  ++it2; }
          // now [it, it2] bracket one detid

          // then prepare to copy the clusters
          typename edmNew::DetSetVector<ClusterType>::FastFiller filler(dsv, detid);
          typename HitType::ClusterRef lastRef, newRef;
          for ( ; it != it2; ++it) { // loop on the detid
              // first check if we need to clone the hit
              if (it->clusterRef() != lastRef) { 
                  lastRef = it->clusterRef();
                  // clone cluster
                  filler.push_back( *lastRef );  
                  // make new ref
                  newRef = typename HitType::ClusterRef( refprod, clusters++ );
              } 
              // then fixup the reference
              it->rekey( *selHits_, newRef );

          } // end of the loop on a single detid

      } // end of the loop on all clusters

      clusterRecords.clear();
  } // end of the function

  //------------------------------------------------------------------
  //!  Put tracks, track extras and hits+clusters into the event.
  //------------------------------------------------------------------
  edm::OrphanHandle<reco::TrackCollection> 
  TrackCollectionStoreManager::
  put( edm::Event & evt ) {
    edm::OrphanHandle<reco::TrackCollection> 
      h = evt.put( selTracks_ );
    evt.put( selTrackExtras_ );
    evt.put( selHits_ );
    evt.put( selStripClusters_ );
    evt.put( selPixelClusters_ );
    return h; 
  }
  

} // end of namespace helper

