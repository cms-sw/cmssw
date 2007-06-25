#ifndef RecoAlgos_ElectonSelector_h
#define RecoAlgos_ElectronSelector_h
/** \class ElectronSelector
 *
 * selects a subset of an electron collection. Also clones
 * all referenced objects
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.8 $
 *
 * $Id: ElectronSelector.h,v 1.8 2006/12/07 11:28:31 llista Exp $
 *
 */

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"

namespace helper {
  struct ElectronCollectionStoreManager {
    typedef reco::ElectronCollection collection;
    ElectronCollectionStoreManager() :
      selElectrons_( new reco::ElectronCollection ),
      selSuperClusters_( new reco::SuperClusterCollection ),
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
      ElectronRefProd rElectrons = evt.template getRefBeforePut<ElectronCollection>();      
      SuperClusterRefProd rSuperClusters = evt.template getRefBeforePut<SuperClusterCollection>();      
      size_t idx = 0, tidx = 0, hidx = 0;
      for( I i = begin; i != end; ++ i ) {
	const Electron & ele = * * i;
	selElectrons_->push_back( Electron( ele ) );
	selElectrons_->back().setTrack( TrackRef( rTracks, idx ) );
	selElectrons_->back().setSuperCluster( SuperClusterRef( rSuperClusters, idx ++ ) );
	selSuperClusters_->push_back( SuperCluster( * ( ele.superCluster() ) ) );
	TrackRef trkRef = ele.track();
	if ( trkRef.isNonnull() ) {
	  selTracks_->push_back( Track( * trkRef ) );
  	  Track & trk = selTracks_->back();
	  selTrackExtras_->push_back( TrackExtra( trk.outerPosition(), trk.outerMomentum(), trk.outerOk(),
						  trk.innerPosition(), trk.innerMomentum(), trk.innerOk(),
						  trk.outerStateCovariance(), trk.outerDetId(),
						  trk.innerStateCovariance(), trk.innerDetId() ) );
  	  TrackExtra & tx = selTrackExtras_->back();
	  for( trackingRecHit_iterator hit = trk.recHitsBegin(); hit != trk.recHitsEnd(); ++ hit ) {
	    selHits_->push_back( (*hit)->clone() );
	    tx.add( TrackingRecHitRef( rHits, hidx ++ ) );
	  }
 	  trk.setExtra( TrackExtraRef( rTrackExtras, tidx ++ ) ); 
	} 
      }
    }
    edm::OrphanHandle<reco::ElectronCollection> put( edm::Event & evt ) {
      edm::OrphanHandle<reco::ElectronCollection> h = evt.put( selElectrons_ );
      evt.put( selSuperClusters_ );
      evt.put( selTracks_ );
      evt.put( selTrackExtras_ );
      evt.put( selHits_ );
      return h;
    }
    size_t size() const { return selElectrons_->size(); }
  private:
    std::auto_ptr<reco::ElectronCollection> selElectrons_;
    std::auto_ptr<reco::SuperClusterCollection> selSuperClusters_;
    std::auto_ptr<reco::TrackCollection> selTracks_;
    std::auto_ptr<reco::TrackExtraCollection> selTrackExtras_;
    std::auto_ptr<TrackingRecHitCollection> selHits_;
  };

  class ElectronSelectorBase : public edm::EDFilter {
  public:
    ElectronSelectorBase( const edm::ParameterSet & cfg ) {
      std::string alias( cfg.getParameter<std::string>( "@module_label" ) );
      produces<reco::ElectronCollection>().setBranchAlias( alias + "Electrons" );
      produces<reco::SuperClusterCollection>().setBranchAlias( alias + "SuperClusters" );
      produces<reco::TrackCollection>().setBranchAlias( alias + "Tracks" );
      produces<reco::TrackExtraCollection>().setBranchAlias( alias + "TrackExtras" );
      produces<TrackingRecHitCollection>().setBranchAlias( alias + "RecHits" );
    }
   };


  template<>
  struct StoreManagerTrait<reco::ElectronCollection> {
    typedef ElectronCollectionStoreManager type;
    typedef ElectronSelectorBase base;
  };

}

#endif
