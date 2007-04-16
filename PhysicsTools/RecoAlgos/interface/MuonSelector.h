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
 * \version $Revision: 1.1 $
 *
 * $Id: MuonSelector.h,v 1.0 2007/03/22 12:22:11 jfernan2 Exp $
 *
 */
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"

namespace helper {
  struct MuonCollectionStoreManager {
    typedef reco::MuonCollection collection;
    MuonCollectionStoreManager() :
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

      TrackingRecHitRefProd rHits = evt.template getRefBeforePut<TrackingRecHitCollection>();
      TrackExtraRefProd rTrackExtras = evt.template getRefBeforePut<TrackExtraCollection>();
      TrackRefProd rTracks = evt.template getRefBeforePut<TrackCollection>();      

      MuonRefProd rMuons = evt.template getRefBeforePut<MuonCollection>();      

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
//	id++;
//	selMuons_->back().setTrack( trkRef);

	selTracks_->push_back(Track( *trkRef) );

	Track & trk= selTracks_->back();

	selTracksExtras_->push_back( TrackExtra( trk.outerPosition(), trk.outerMomentum(), trk.outerOk(),
						trk.innerPosition(), trk.innerMomentum(), trk.innerOk(),
						trk.outerStateCovariance(), trk.outerDetId(),
						trk.innerStateCovariance(), trk.innerDetId() ) );

	TrackExtra & tx = selTracksExtras_->back();

	for( trackingRecHit_iterator hit = trk.recHitsBegin(); hit != trk.recHitsEnd(); ++ hit ) {
  	  selTracksHits_->push_back( (*hit)->clone() );
	  tx.add( TrackingRecHitRef( rHits, hidx ++ ) );
	}
	
	trk.setExtra( TrackExtraRef( rTrackExtras, idx ++ ) );

	}

	// global Muon Track	
	selMuons_->back().setCombined( TrackRef( rTracks, igbd ++ ) );
	trkRef = mu.combinedMuon();
	if(trkRef.isNonnull()){
//	igbd++;
//	selMuons_->back().setCombined(trkRef);
	selGlobalMuonTracks_->push_back(Track( *trkRef) );
	Track & trk = selGlobalMuonTracks_->back();
		
	selGlobalMuonTracksExtras_->push_back( TrackExtra( trk.outerPosition(), trk.outerMomentum(), trk.outerOk(),
						trk.innerPosition(), trk.innerMomentum(), trk.innerOk(),
						trk.outerStateCovariance(), trk.outerDetId(),
						trk.innerStateCovariance(), trk.innerDetId() ) );
	TrackExtra & tx = selGlobalMuonTracksExtras_->back();
	for( trackingRecHit_iterator hit = trk.recHitsBegin(); hit != trk.recHitsEnd(); ++ hit ) {
	  selGlobalMuonTracksHits_->push_back( (*hit)->clone() );
	  tx.add( TrackingRecHitRef( rHits, higbdx ++ ) );
	}
	trk.setExtra( TrackExtraRef( rTrackExtras, igbdx ++ ) );

	}

	// stand alone Muon Track	
	selMuons_->back().setStandAlone( TrackRef( rTracks, isad ++ ) );
	trkRef = mu.standAloneMuon();
	if(trkRef.isNonnull()){
//	isad++;
//	selMuons_->back().setStandAlone( trkRef );
	selStandAloneTracks_->push_back(Track( *trkRef) );
	Track & trk = selStandAloneTracks_->back();
		
	selStandAloneTracksExtras_->push_back( TrackExtra( trk.outerPosition(), trk.outerMomentum(), trk.outerOk(),
						trk.innerPosition(), trk.innerMomentum(), trk.innerOk(),
						trk.outerStateCovariance(), trk.outerDetId(),
						trk.innerStateCovariance(), trk.innerDetId() ) );
	TrackExtra & tx = selStandAloneTracksExtras_->back();
	for( trackingRecHit_iterator hit = trk.recHitsBegin(); hit != trk.recHitsEnd(); ++ hit ) {
	  selStandAloneTracksHits_->push_back( (*hit)->clone() );
	  tx.add( TrackingRecHitRef( rHits, hisadx ++ ) );
	}
	trk.setExtra( TrackExtraRef( rTrackExtras, isadx ++ ) );

	}
      }
//     edm::LogDebug("AlignmentMuonSelector") 
/*	std::cout<<"##################"<<std::endl;
//      edm::LogDebug("AlignmentMuonSelector") 
	std::cout<<"c="<< c<<" id="<< id <<" igbd="<<igbd<<" isad="<< isad<<"
	idx="<< idx<< " igdbx="<< igbdx<< " isadx="<< isadx <<" hidx=" <<hidx<<
	" higbdx="<< higbdx<< " hisadx="<<hisadx<<std::endl;*/
    }
    
    edm::OrphanHandle<reco::MuonCollection> put( edm::Event & evt ) {
      evt.put( selTracks_ );
      evt.put( selTracksExtras_ );
      evt.put( selTracksHits_ );
      edm::OrphanHandle<reco::MuonCollection> h =       evt.put( selMuons_ );
//      edm::OrphanHandle<reco::MuonCollection> h = evt.put( selMuons_ , "SelectedMuons");
/*      evt.put( selTracks_ , "TrackerOnly");
      evt.put( selTracksExtras_ , "TrackerOnly");
      evt.put( selTracksHits_ ,"TrackerOnly");*/
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

      produces<reco::MuonCollection>().setBranchAlias( alias + "Muons" );
      produces<reco::TrackCollection>().setBranchAlias( alias + "TrackerOnlyTracks" );
      produces<reco::TrackExtraCollection>().setBranchAlias( alias + "TrackerOnlyExtras" );
      produces<TrackingRecHitCollection>().setBranchAlias( alias + "TrackerOnlyHits" );
/*      produces<reco::MuonCollection>("SelectedMuons").setBranchAlias( alias + "Muons" );
      produces<reco::TrackCollection>("TrackerOnly").setBranchAlias( alias + "TrackerOnlyTracks" );
      produces<reco::TrackExtraCollection>("TrackerOnly").setBranchAlias( alias + "TrackerOnlyExtras" );
      produces<TrackingRecHitCollection>("TrackerOnly").setBranchAlias( alias + "TrackerOnlyHits" );*/
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
