#ifndef RecoAlgos_GsfElectronSelector_h
#define RecoAlgos_GsfElectronSelector_h
/** \class GsfElectronSelector
 *
 * selects a subset of an electron collection. Also clones
 * all referenced objects
 * 
 * \author Luca Lista, INFN
 *
 * \version $Revision: 1.5 $
 *
 * $Id: GsfElectronSelector.h,v 1.5 2010/09/27 07:48:51 gpetrucc Exp $
 *
 */

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronCore.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtra.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"

namespace helper {
  struct GsfElectronCollectionStoreManager {
    typedef reco::GsfElectronCollection collection;
    GsfElectronCollectionStoreManager(const edm::Handle<reco::GsfElectronCollection>&) :
      selElectrons_( new reco::GsfElectronCollection ),
      selElectronCores_ (new reco::GsfElectronCoreCollection ) ,
      selSuperClusters_( new reco::SuperClusterCollection ),
      selTracks_( new reco::GsfTrackCollection ),
      selTrackExtras_( new reco::TrackExtraCollection ),
      selGsfTrackExtras_( new reco::GsfTrackExtraCollection ),
      selHits_( new TrackingRecHitCollection ) {
    }
    template<typename I>
    void cloneAndStore( const I & begin, const I & end, edm::Event & evt ) {
      using namespace reco;
      TrackingRecHitRefProd rHits = evt.template getRefBeforePut<TrackingRecHitCollection>();
      TrackExtraRefProd rTrackExtras = evt.template getRefBeforePut<TrackExtraCollection>();
      GsfTrackExtraRefProd rGsfTrackExtras = evt.template getRefBeforePut<GsfTrackExtraCollection>();
      GsfTrackRefProd rTracks = evt.template getRefBeforePut<GsfTrackCollection>();      
      GsfElectronCoreRefProd rElectronCores = evt.template getRefBeforePut<GsfElectronCoreCollection>();      
      GsfElectronRefProd rElectrons = evt.template getRefBeforePut<GsfElectronCollection>();      
      SuperClusterRefProd rSuperClusters = evt.template getRefBeforePut<SuperClusterCollection>();      
      size_t idx = 0, tidx = 0, hidx = 0;
      for( I i = begin; i != end; ++ i ) {
	const GsfElectron & ele = * * i;
	selElectronCores_->push_back( GsfElectronCore( *(ele.core()) ) );
	selElectronCores_->back().setGsfTrack( GsfTrackRef( rTracks, idx ) );
	selElectronCores_->back().setSuperCluster( SuperClusterRef( rSuperClusters, idx ) );
	selSuperClusters_->push_back( SuperCluster( * ( ele.superCluster() ) ) );
        selElectrons_->push_back(GsfElectron(ele,GsfElectronCoreRef(rElectronCores,idx++),CaloClusterPtr(),TrackRef(),TrackBaseRef(),GsfTrackRefVector()));
	GsfTrackRef trkRef = ele.gsfTrack();
	if ( trkRef.isNonnull() ) {
	  selTracks_->push_back( GsfTrack( * trkRef ) );
  	  GsfTrack & trk = selTracks_->back();
	  selTrackExtras_->push_back( TrackExtra( trk.outerPosition(), trk.outerMomentum(), trk.outerOk(),
						  trk.innerPosition(), trk.innerMomentum(), trk.innerOk(),
						  trk.outerStateCovariance(), trk.outerDetId(),
						  trk.innerStateCovariance(), trk.innerDetId(),
						  trk.seedDirection() ) );
	  selGsfTrackExtras_->push_back( GsfTrackExtra( *(trk.gsfExtra()) ) );
  	  TrackExtra & tx = selTrackExtras_->back();
          unsigned int nHitsToAdd = 0;
	  for( trackingRecHit_iterator hit = trk.recHitsBegin(); hit != trk.recHitsEnd(); ++ hit ) {
	    selHits_->push_back( (*hit)->clone() );
            ++nHitsToAdd;
	  }
          tx.setHits( rHits, hidx, nHitsToAdd );
          tx.setTrajParams(trk.extra()->trajParams(),trk.extra()->chi2sX5());
          assert(tx.trajParams().size()==tx.recHitsSize());
          hidx += nHitsToAdd;
 	  trk.setGsfExtra( GsfTrackExtraRef( rGsfTrackExtras, tidx ) ); 
 	  trk.setExtra( TrackExtraRef( rTrackExtras, tidx ++ ) ); 
	} 
      }
    }

    edm::OrphanHandle<reco::GsfElectronCollection> put( edm::Event & evt ) {
      edm::OrphanHandle<reco::GsfElectronCollection> h = evt.put(std::move(selElectrons_));
      evt.put(std::move(selElectronCores_ ));
      evt.put(std::move(selSuperClusters_ ));
      evt.put(std::move(selTracks_ ));
      evt.put(std::move(selTrackExtras_ ));
      evt.put(std::move(selGsfTrackExtras_ ));
      evt.put(std::move(selHits_ ));
      return h;
    }

    size_t size() const { return selElectrons_->size(); }
  private:
    std::unique_ptr<reco::GsfElectronCollection> selElectrons_;
    std::unique_ptr<reco::GsfElectronCoreCollection> selElectronCores_;
    std::unique_ptr<reco::SuperClusterCollection> selSuperClusters_;
    std::unique_ptr<reco::GsfTrackCollection> selTracks_;
    std::unique_ptr<reco::TrackExtraCollection> selTrackExtras_;
    std::unique_ptr<reco::GsfTrackExtraCollection> selGsfTrackExtras_;
    std::unique_ptr<TrackingRecHitCollection> selHits_;
  };

  class GsfElectronSelectorBase : public edm::EDFilter {
  public:
    GsfElectronSelectorBase( const edm::ParameterSet & cfg ) {
      std::string alias( cfg.getParameter<std::string>( "@module_label" ) );
      produces<reco::GsfElectronCollection>().setBranchAlias( alias + "GsfElectrons" );
      produces<reco::GsfElectronCoreCollection>().setBranchAlias( alias + "GsfElectronCores" );
      produces<reco::SuperClusterCollection>().setBranchAlias( alias + "SuperClusters" );
      produces<reco::GsfTrackCollection>().setBranchAlias( alias + "GsfTracks" );
      produces<reco::GsfTrackExtraCollection>().setBranchAlias( alias + "GsfTrackExtras" );
      produces<reco::TrackExtraCollection>().setBranchAlias( alias + "TrackExtras" );
      produces<TrackingRecHitCollection>().setBranchAlias( alias + "RecHits" );
    }
  };
  
  template<>
  struct StoreManagerTrait<reco::GsfElectronCollection> {
    typedef GsfElectronCollectionStoreManager type;
    typedef GsfElectronSelectorBase base;
  };
}

#endif
