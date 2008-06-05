#include "PhysicsTools/RecoAlgos/interface/TrackSelector.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"

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
    idx_(0), hidx_(0), scidx_(), pcidx_()
  {
  }
  
  
  
  //------------------------------------------------------------------
  //!  Process a single track.  THIS IS WHERE ALL THE ACTION HAPPENS!
  //------------------------------------------------------------------
  void 
  TrackCollectionStoreManager::
  processTrack( const Track & trk ) 
  {
    selTracks_->push_back( Track( trk ) );
    selTracks_->back().setExtra( TrackExtraRef( rTrackExtras_, idx_ ++ ) );
    selTrackExtras_->push_back( TrackExtra( trk.outerPosition(), trk.outerMomentum(), trk.outerOk(),
					    trk.innerPosition(), trk.innerMomentum(), trk.innerOk(),
					    trk.outerStateCovariance(), trk.outerDetId(),
					    trk.innerStateCovariance(), trk.innerDetId(),
					    trk.seedDirection() ) );
    TrackExtra & tx = selTrackExtras_->back();
    for( trackingRecHit_iterator hit = trk.recHitsBegin(); hit != trk.recHitsEnd(); ++ hit ) {
      selHits_->push_back( (*hit)->clone() );
      TrackingRecHit * newHit = & (selHits_->back());
      tx.add( TrackingRecHitRef( rHits_, hidx_ ++ ) );
      //
      //--- New: copy strip or pixel cluster.  This is a bit tricky since
      //--- the TrackingRecHit could be either a SiStripRecHit of some kind
      //--- or the SiPixelRecHit.
      
      const DetId detId( (*hit)->geographicalId() );
      if (detId.det() != DetId::Tracker) {
	// FIXME: Throw an exception, or simply return with error status.
	// FIXME: Could a reco::Track even have non-tracker hits???
	assert(0);
      }

      //--- Figure out which kind of hit this is.      
      //    (Note: hit is an iterator to a vector of RecHit pointers, so
      //     *hit is a pointer to a hit.)
      //
      if (detId.subdetId() == PixelSubdetector::PixelBarrel || 
	  detId.subdetId() == PixelSubdetector::PixelEndcap ) {
	// must be a Pixel hit
	const SiPixelRecHit * pixHit = dynamic_cast<const SiPixelRecHit*>( &**hit );
	if (!pixHit) {
	  assert(0);   	  // FIXME: throw a fatal exception
	}
	//--- Get the pixel cluster, clone it, save a ref.  
	//--- In edmNew::DSV, we need to use the FastFiller
	//--- to add a copy of the cluster to a new DSV<SiPixelCluster>
	DetId detid = pixHit->geographicalId();
	edmNew::DetSetVector<SiPixelCluster>::FastFiller pixFF( *selPixelClusters_, detid );

	// Get the cluster. (cluster() returns an edm::Ref<edm::DetSetVector<SiPixelCluster>, SiPixelCluster >  
	const SiPixelCluster * pixCl = &* (pixHit->cluster());
	pixFF.push_back( *pixCl );

	// Create a persistent edm::Ref to the cluster
	edm::Ref< edmNew::DetSetVector<SiPixelCluster>, SiPixelCluster > 
	  pixClRef( rPixelClusters_ , pcidx_ ++ );

	// We must cast since setClusterRef() is not in the base class
	SiPixelRecHit * newPixHit = dynamic_cast<SiPixelRecHit*>( newHit );
	newPixHit->setClusterRef( pixClRef );
      } 
      //
      else { 
        // should be SiStrip now
	const SiStripRecHit2D * sHit1 
	  = dynamic_cast<const SiStripRecHit2D*>( &**hit );
	if (sHit1) {
	  //
	  //--- Get the strip cluster, clone it, save a ref.  
	  //--- In edmNew::DSV, we need to use the FastFiller
	  //--- to add a copy of the cluster to a new DSV<SiStripCluster>
	  DetId detid = sHit1->geographicalId();
	  edmNew::DetSetVector<SiStripCluster>::FastFiller pixFF( *selStripClusters_, detid );
	  
	  // Get the cluster. (cluster() returns an edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster >  
	  const SiStripCluster * strCl = &* (sHit1->cluster());
	  pixFF.push_back( *strCl );
	  
	  // Create a persistent edm::Ref to the cluster
	  edm::Ref<edmNew::DetSetVector<SiStripCluster>,SiStripCluster > 
	    strClRef( rStripClusters_ , scidx_ ++ );

	  // We must cast since setClusterRef() is not in the base class
	  SiStripRecHit2D * newStrHit = dynamic_cast<SiStripRecHit2D*>( newHit );
	  newStrHit->setClusterRef( strClRef );
	  //
 	  continue;       // go to the next hit on the track
	}
	const SiStripMatchedRecHit2D * sHit2 
	  = dynamic_cast<const SiStripMatchedRecHit2D*>( &**hit ); 
	if (sHit2) {
	  // &&& get cluster and clone
	  continue;       // go to the next hit on the track
	}
	const ProjectedSiStripRecHit2D * sHit3 
	  = dynamic_cast<const ProjectedSiStripRecHit2D*>( &**hit );
	if (sHit3) {
	  // &&& get cluster and clone
	  continue;       // go to the next hit on the track
	}
	//--- If we are here, we are in trouble
	edm::LogError("UnkownType") 
	  << "@SUB=AlignmentTrackSelector::isHit2D"
	  << "Tracker hit not in pixel and neither SiStripRecHit2D nor "
	  << "SiStripMatchedRecHit2D nor ProjectedSiStripRecHit2D.";
	// &&& throw an exception?
	return;
      }
    
      //--- What needs to be done is to retrieve clusters from the 
      //--- TrackingRecHit.
      std::cout << hidx_ << " " << scidx_ << " " << pcidx_ << std::endl;

    } // end of for loop
  }
  

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

