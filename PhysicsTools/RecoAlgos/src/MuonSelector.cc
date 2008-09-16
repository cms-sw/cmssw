#include "PhysicsTools/RecoAlgos/interface/MuonSelector.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"

#include <DataFormats/SiStripDetId/interface/SiStripDetId.h>

template<typename RecHitType, typename ClusterRefType>
void 
helper::MuonCollectionStoreManager::ClusterHitRecord<RecHitType,ClusterRefType>::
    rekey(const ClusterRefType &newRef) const  
{
    //std::cout << "Rekeying hit with vector " << hitVector_ << ", index " << index_ << ", detid = " << detid_ << std::endl;
    TrackingRecHit & genericHit = (*hitVector_)[index_]; 
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
    MuonCollectionStoreManager::
    MuonCollectionStoreManager(const edm::Handle<reco::MuonCollection>&) 
      :
      selMuons_( new reco::MuonCollection ),
      selTracks_( new reco::TrackCollection ),
      selTracksExtras_( new reco::TrackExtraCollection ),
      selTracksHits_( new TrackingRecHitCollection ),
      selGlobalMuonTracks_( new reco::TrackCollection ),
      selGlobalMuonTracksExtras_( new reco::TrackExtraCollection ),    
      selGlobalMuonTracksHits_( new TrackingRecHitCollection ),
      selStandAloneTracks_( new reco::TrackCollection ),
      selStandAloneTracksExtras_( new reco::TrackExtraCollection ),
      selStandAloneTracksHits_( new TrackingRecHitCollection ),     
      selStripClusters_( new edmNew::DetSetVector<SiStripCluster> ),
      selPixelClusters_( new edmNew::DetSetVector<SiPixelCluster> ),
      rMuons_(),
      rTracks_(), rTrackExtras_(), rHits_(), rStripClusters_(), rPixelClusters_(),
      rGBTracks_(), rGBTrackExtras_(), rGBHits_(),
      rSATracks_(), rSATrackExtras_(), rSAHits_(),
      id_(0), igbd_(0), isad_(0), idx_(0), igbdx_(0),
      isadx_(0), hidx_(0), higbdx_(0), hisadx_(0),
      cloneClusters_ (true),clustersOK_ (true)
    {
    }

  
  //------------------------------------------------------------------
  //!  Process a single muon.  
  //------------------------------------------------------------------
  void 
  MuonCollectionStoreManager::
  processMuon( const Muon & mu ) 
  {
	selMuons_->push_back( Muon( mu ) );
	// only tracker Muon Track	
	selMuons_->back().setInnerTrack( TrackRef( rTracks_, id_ ++ ) );
	TrackRef trkRef = mu.track();
	if(trkRef.isNonnull()){

	selTracks_->push_back(Track( *trkRef) );

	Track & trk= selTracks_->back();

	selTracksExtras_->push_back( TrackExtra( trk.outerPosition(), trk.outerMomentum(), trk.outerOk(),
						 trk.innerPosition(), trk.innerMomentum(), trk.innerOk(),
						 trk.outerStateCovariance(), trk.outerDetId(),
						 trk.innerStateCovariance(), trk.innerDetId(),
						 trk.seedDirection() ) );

	TrackExtra & tx = selTracksExtras_->back();

	for( trackingRecHit_iterator hit = trk.recHitsBegin(); hit != trk.recHitsEnd(); ++ hit ) {
  	  selTracksHits_->push_back( (*hit)->clone() );
          TrackingRecHit * newHit = & (selTracksHits_->back());
	  tx.add( TrackingRecHitRef( rHits_, hidx_ ++ ) );
          if (cloneClusters()) processHit( newHit, *selTracksHits_ );
	} // end of for loop over tracking rec hits on this track
	
	trk.setExtra( TrackExtraRef( rTrackExtras_, idx_ ++ ) );

	}// TO trkRef.isNonnull

	// global Muon Track	
	selMuons_->back().setGlobalTrack( TrackRef( rGBTracks_, igbd_ ++ ) );
	trkRef = mu.combinedMuon();
	if(trkRef.isNonnull()){
	selGlobalMuonTracks_->push_back(Track( *trkRef) );
	Track & trk = selGlobalMuonTracks_->back();
		
	selGlobalMuonTracksExtras_->push_back( TrackExtra( trk.outerPosition(), trk.outerMomentum(), trk.outerOk(),
						trk.innerPosition(), trk.innerMomentum(), trk.innerOk(),
						trk.outerStateCovariance(), trk.outerDetId(),
						trk.innerStateCovariance(), trk.innerDetId(), trk.seedDirection() ) );
	TrackExtra & tx = selGlobalMuonTracksExtras_->back();
	for( trackingRecHit_iterator hit = trk.recHitsBegin(); hit != trk.recHitsEnd(); ++ hit ) {
            selGlobalMuonTracksHits_->push_back( (*hit)->clone() );
            TrackingRecHit * newHit = & (selGlobalMuonTracksHits_->back()); 
            tx.add( TrackingRecHitRef( rGBHits_, higbdx_ ++ ) );
            if (cloneClusters()) processHit( newHit, *selGlobalMuonTracksHits_ );
	}
	trk.setExtra( TrackExtraRef( rGBTrackExtras_, igbdx_ ++ ) );

	} // GB trkRef.isNonnull()

	// stand alone Muon Track	
	selMuons_->back().setOuterTrack( TrackRef( rSATracks_, isad_ ++ ) );
	trkRef = mu.standAloneMuon();
	if(trkRef.isNonnull()){
	selStandAloneTracks_->push_back(Track( *trkRef) );
	Track & trk = selStandAloneTracks_->back();
		
	selStandAloneTracksExtras_->push_back( TrackExtra( trk.outerPosition(), trk.outerMomentum(), trk.outerOk(),
						trk.innerPosition(), trk.innerMomentum(), trk.innerOk(),
						trk.outerStateCovariance(), trk.outerDetId(),
						trk.innerStateCovariance(), trk.innerDetId(), trk.seedDirection() ) );
	TrackExtra & tx = selStandAloneTracksExtras_->back();
	for( trackingRecHit_iterator hit = trk.recHitsBegin(); hit != trk.recHitsEnd(); ++ hit ) {
	  selStandAloneTracksHits_->push_back( (*hit)->clone() );
	  tx.add( TrackingRecHitRef( rSAHits_, hisadx_ ++ ) );
	}
	trk.setExtra( TrackExtraRef( rSATrackExtras_, isadx_ ++ ) );

	} // SA trkRef.isNonnull()
  }// end of track, and function

  //------------------------------------------------------------------
  //!  Process a single hit.  
  //------------------------------------------------------------------
  void
  MuonCollectionStoreManager::
  processHit( const TrackingRecHit * hit, edm::OwnVector<TrackingRecHit> &hits ) {
        //--- Skip the rest for this hit if we don't want to clone the cluster.
        //--- The copy constructer in the rec hit will copy the link properly.
        //

        //std::cout << "|   I'm cloing clusters, hit vector = " << (&hits) << std::endl;

        const DetId detId( hit->geographicalId() );
        if (hit->isValid() && (detId.det() == DetId::Tracker)) {
            //std::cout << "|   It is a tracker hit" << std::endl;

            const std::type_info & hit_type = typeid(*hit);
            if (hit_type == typeid(SiPixelRecHit)) {
                //std::cout << "|  It is a Pixel hit !!" << std::endl;
                pixelClusterRecords_.push_back( PixelClusterHitRecord( static_cast<const SiPixelRecHit &>(*hit), &hits, hits.size() - 1) );
            } else if (hit_type == typeid(SiStripRecHit2D)) {
                //std::cout << "|   It is a SiStripRecHit2D hit !!" << std::endl;
                stripClusterRecords_.push_back( StripClusterHitRecord( static_cast<const SiStripRecHit2D &>(*hit), &hits, hits.size() - 1) );
            } else if (hit_type == typeid(SiStripMatchedRecHit2D)) {      
                //std::cout << "|   It is a SiStripMatchedRecHit2D hit !!" << std::endl;
                const SiStripMatchedRecHit2D & mhit = static_cast<const SiStripMatchedRecHit2D &>(*hit);
                stripClusterRecords_.push_back( StripClusterHitRecord( *mhit.monoHit()  , &hits, hits.size() - 1) );
                stripClusterRecords_.push_back( StripClusterHitRecord( *mhit.stereoHit(), &hits, hits.size() - 1) );
            } else if (hit_type == typeid(ProjectedSiStripRecHit2D)) {
                //std::cout << "|   It is a ProjectedSiStripRecHit2D hit !!" << std::endl;
                const ProjectedSiStripRecHit2D & phit = static_cast<const ProjectedSiStripRecHit2D &>(*hit);
                stripClusterRecords_.push_back( StripClusterHitRecord( phit.originalHit(), &hits, hits.size() - 1) );
            } else {
                //std::cout << "|   It is a " << hit_type.name() << " hit !?" << std::endl;
                // do nothing. We might end up here for FastSim hits.
            } // end 'switch' on hit type
        } // end if it was a tracker hit

  }
 
  void
  MuonCollectionStoreManager::
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
  MuonCollectionStoreManager::
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
                  if(lastRef.isAvailable()){
                  filler.push_back( *lastRef );  
                  // make new ref
                  newRef = typename HitType::ClusterRef( refprod, clusters++ );
		  }
		  else setClustersOK(false);
		 // to avoid problem with missing clusters refs in 21X 
              } 
              // then fixup the reference
              it->rekey( newRef );

          } // end of the loop on a single detid

      } // end of the loop on all clusters

      clusterRecords.clear();
  } // end of the function


  //------------------------------------------------------------------
  //!  Put Muons, tracks, track extras and hits+clusters into the event.
  //------------------------------------------------------------------
    edm::OrphanHandle<reco::MuonCollection> 
    MuonCollectionStoreManager::
    put( edm::Event & evt ) {
	edm::OrphanHandle<reco::MuonCollection> h;
    if(clustersOK()){
      h = evt.put( selMuons_ , "SelectedMuons");
      evt.put( selTracks_ , "TrackerOnly");
      evt.put( selTracksExtras_ , "TrackerOnly");
      evt.put( selTracksHits_ ,"TrackerOnly");
      evt.put( selGlobalMuonTracks_,"GlobalMuon" );
      evt.put( selGlobalMuonTracksExtras_ ,"GlobalMuon");
      evt.put( selGlobalMuonTracksHits_,"GlobalMuon" );
      evt.put( selStandAloneTracks_ ,"StandAlone");
      evt.put( selStandAloneTracksExtras_ ,"StandAlone");
      evt.put( selStandAloneTracksHits_ ,"StandAlone");
      if (cloneClusters()) {
          evt.put( selStripClusters_ );
          evt.put( selPixelClusters_ );
      }
     }      
     else{
        edm::LogError("MuonSelector")<<"Missing reference from clusters!!!";
        edm::LogError("MuonSelector")<<"No objects will be produced!!!";
     }

      return h; 
     
    }


} // end of namespace helper

      
 
