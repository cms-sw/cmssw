#ifndef RecoAlgos_MuonSelector_h
#define RecoAlgos_MuonSelector_h
/** \class MuonSelector
 *
 * selects a subset of a muon collection and clones
 * Track, TrackExtra parts and RecHits collection
 * for SA, GB and Tracker Only options
 * 
 * \author Javier Fernandez, IFCA
 *
 * \version $Revision: 1.6 $
 *
 * $Id: MuonSelector.h,v 1.6 2007/09/20 18:48:28 llista Exp $
 *
 */
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"

namespace helper {
  struct MuonCollectionStoreManager {
    typedef reco::MuonCollection collection;
    MuonCollectionStoreManager(const edm::Handle<reco::MuonCollection>&) :
      selMuons_( new reco::MuonCollection ),
      selTracks_( new reco::TrackCollection ),
      selTracksExtras_( new reco::TrackExtraCollection ),
      selTracksHits_( new TrackingRecHitCollection ),
      selGlobalMuonTracks_( new reco::TrackCollection ),
      selGlobalMuonTracksExtras_( new reco::TrackExtraCollection ),
      selGlobalMuonTracksHits_( new TrackingRecHitCollection ),
      selStandAloneTracks_( new reco::TrackCollection ),
      selStandAloneTracksExtras_( new reco::TrackExtraCollection ),
      selStandAloneTracksHits_( new TrackingRecHitCollection ) {      
      }
    
    template<typename I>
    void cloneAndStore( const I & begin, const I & end, edm::Event & evt ) {
      using namespace reco;

      TrackingRecHitRefProd rHits = evt.template getRefBeforePut<TrackingRecHitCollection>("TrackerOnly");
      TrackingRecHitRefProd rGBHits = evt.template getRefBeforePut<TrackingRecHitCollection>("GlobalMuon");
      TrackingRecHitRefProd rSAHits = evt.template getRefBeforePut<TrackingRecHitCollection>("StandAlone");
      TrackExtraRefProd rTrackExtras = evt.template getRefBeforePut<TrackExtraCollection>("TrackerOnly");
      TrackExtraRefProd rGBTrackExtras = evt.template getRefBeforePut<TrackExtraCollection>("GlobalMuon");
      TrackExtraRefProd rSATrackExtras = evt.template getRefBeforePut<TrackExtraCollection>("StandAlone");
      TrackRefProd rTracks = evt.template getRefBeforePut<TrackCollection>("TrackerOnly");      
      TrackRefProd rGBTracks = evt.template getRefBeforePut<TrackCollection>("GlobalMuon");      
      TrackRefProd rSATracks = evt.template getRefBeforePut<TrackCollection>("StandAlone");      

      MuonRefProd rMuons = evt.template getRefBeforePut<MuonCollection>("SelectedMuons");      

	int c=0;
      size_t id=0, igbd=0, isad=0, idx = 0, igbdx=0, isadx=0, hidx = 0, higbdx=0, hisadx=0;
      for( I i = begin; i != end; ++ i ) {
	c++;
	const Muon & mu = * * i;
	selMuons_->push_back( Muon( mu ) );
	// only tracker Muon Track	
	selMuons_->back().setTrack( TrackRef( rTracks, id ++ ) );
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
	  tx.add( TrackingRecHitRef( rHits, hidx ++ ) );
	}
	
	trk.setExtra( TrackExtraRef( rTrackExtras, idx ++ ) );

	}

	// global Muon Track	
	selMuons_->back().setCombined( TrackRef( rGBTracks, igbd ++ ) );
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
	  tx.add( TrackingRecHitRef( rGBHits, higbdx ++ ) );
	}
	trk.setExtra( TrackExtraRef( rGBTrackExtras, igbdx ++ ) );

	}

	// stand alone Muon Track	
	selMuons_->back().setStandAlone( TrackRef( rSATracks, isad ++ ) );
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
	  tx.add( TrackingRecHitRef( rSAHits, hisadx ++ ) );
	}
	trk.setExtra( TrackExtraRef( rSATrackExtras, isadx ++ ) );

	}
      }
    }
    
    edm::OrphanHandle<reco::MuonCollection> put( edm::Event & evt ) {
      edm::OrphanHandle<reco::MuonCollection> h = evt.put( selMuons_ , "SelectedMuons");
      evt.put( selTracks_ , "TrackerOnly");
      evt.put( selTracksExtras_ , "TrackerOnly");
      evt.put( selTracksHits_ ,"TrackerOnly");
      evt.put( selGlobalMuonTracks_,"GlobalMuon" );
      evt.put( selGlobalMuonTracksExtras_ ,"GlobalMuon");
      evt.put( selGlobalMuonTracksHits_,"GlobalMuon" );
      evt.put( selStandAloneTracks_ ,"StandAlone");
      evt.put( selStandAloneTracksExtras_ ,"StandAlone");
      evt.put( selStandAloneTracksHits_ ,"StandAlone");
      return h; 
    }
    size_t size() const { return selMuons_->size(); }
  private:
    std::auto_ptr<reco::MuonCollection> selMuons_;
    std::auto_ptr<reco::TrackCollection> selTracks_;
    std::auto_ptr<reco::TrackExtraCollection> selTracksExtras_;
    std::auto_ptr<TrackingRecHitCollection> selTracksHits_;
    std::auto_ptr<reco::TrackCollection> selGlobalMuonTracks_;
    std::auto_ptr<reco::TrackExtraCollection> selGlobalMuonTracksExtras_;
    std::auto_ptr<TrackingRecHitCollection> selGlobalMuonTracksHits_;
    std::auto_ptr<reco::TrackCollection> selStandAloneTracks_;
    std::auto_ptr<reco::TrackExtraCollection> selStandAloneTracksExtras_;
    std::auto_ptr<TrackingRecHitCollection> selStandAloneTracksHits_;
  };

  class MuonSelectorBase : public edm::EDFilter {
  public:
    MuonSelectorBase( const edm::ParameterSet & cfg ) {
      std::string alias( cfg.getParameter<std::string>( "@module_label" ) );


      produces<reco::MuonCollection>("SelectedMuons").setBranchAlias( alias + "SelectedMuons" );
      produces<reco::TrackCollection>("TrackerOnly").setBranchAlias( alias + "TrackerOnlyTracks" );
      produces<reco::TrackExtraCollection>("TrackerOnly").setBranchAlias( alias + "TrackerOnlyExtras" );
      produces<TrackingRecHitCollection>("TrackerOnly").setBranchAlias( alias + "TrackerOnlyHits" );
      produces<reco::TrackCollection>("GlobalMuon").setBranchAlias( alias + "GlobalMuonTracks" );
      produces<reco::TrackExtraCollection>("GlobalMuon").setBranchAlias( alias + "GlobalMuonExtras" );
      produces<TrackingRecHitCollection>("GlobalMuon").setBranchAlias( alias + "GlobalMuonHits" );
      produces<reco::TrackCollection>("StandAlone").setBranchAlias( alias + "StandAloneTracks" );
      produces<reco::TrackExtraCollection>("StandAlone").setBranchAlias( alias + "StandAloneExtras" );
      produces<TrackingRecHitCollection>("StandAlone").setBranchAlias( alias + "StandAloneHits" );

    }
   };

  template<>
  struct StoreManagerTrait<reco::MuonCollection> {
    typedef MuonCollectionStoreManager type;
    typedef MuonSelectorBase base;
  };

}

#endif
