#ifndef RecoAlgos_MuonSelector_h
#define RecoAlgos_MuonSelector_h
/** \class MuonSelector
 *
 * selects a subset of a muon collection and clones
 * Track, TrackExtra, RecHits and 'used tracker cluster' collections
 * for SA, GB and Tracker Only options
 * 
 * \author Javier Fernandez, Uniovi
 *
 * \version $Revision: 1.1 $
 *
 * $Id: MuonSelector.h,v 1.1 2009/03/04 13:11:28 llista Exp $
 *
 */
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "CommonTools/RecoAlgos/interface/ClusterStorer.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"

namespace helper {
  struct MuonCollectionStoreManager {
   public:
    typedef reco::MuonCollection collection;

    MuonCollectionStoreManager(const edm::Handle<reco::MuonCollection>&) ;
    
    //------------------------------------------------------------------
    //!  Use these to turn off/on the cloning of clusters.  The default
    //!  is to clone them.  To not clone (and save space in a quick local
    //!  job, do:  
    //!              setCloneClusters(false);
    //------------------------------------------------------------------
    inline bool cloneClusters() {return cloneClusters_ ; } 
    inline void setCloneClusters(bool w) { cloneClusters_ = w; }

    //------------------------------------------------------------------
    //!  Put tracks, track extras and hits+clusters into the event.
    //------------------------------------------------------------------
    edm::OrphanHandle<reco::MuonCollection> put( edm::Event & evt );

    //------------------------------------------------------------------
    //!  Get the size.
    //------------------------------------------------------------------
    inline size_t size() const { return selMuons_->size(); }
    

    //------------------------------------------------------------------
    //! \brief Method to clone tracks, track extras and their hits and clusters.
    //! typename I = this is an interator over a Muon collection, **I needs
    //! to dereference into a Muon.
    //------------------------------------------------------------------
    template<typename I>
    void cloneAndStore( const I & begin, const I & end, edm::Event & evt ) ;
    
  private:
     //--- Collections to store:
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
    std::auto_ptr< edmNew::DetSetVector<SiStripCluster> >  selStripClusters_;
    std::auto_ptr< edmNew::DetSetVector<SiPixelCluster> >  selPixelClusters_;

    reco::MuonRefProd rMuons_;      
    reco::TrackRefProd rTracks_;      
    reco::TrackExtraRefProd rTrackExtras_;
    TrackingRecHitRefProd rHits_;

    reco::TrackRefProd rGBTracks_;      
    reco::TrackExtraRefProd rGBTrackExtras_;
    TrackingRecHitRefProd rGBHits_;
    
    reco::TrackRefProd rSATracks_;      
    reco::TrackExtraRefProd rSATrackExtras_;
    TrackingRecHitRefProd rSAHits_;

    /// Helper to treat copies of selected clusters
    ///  and make the hits refer to the output cluster collections:
    ClusterStorer clusterStorer_;

    //--- Indices into collections handled with RefProd
    size_t id_, igbd_, isad_, idx_, igbdx_, isadx_, hidx_, higbdx_, hisadx_;

    //--- Switches 
    bool   cloneClusters_ ;  //!< Clone clusters, or not?  Default: true.
    
    //--- Methods
    //------------------------------------------------------------------
    //!  Process a single muon.  
    //------------------------------------------------------------------
    void processMuon( const reco::Muon & mu );

    bool clusterRefsOK(const reco::Track &track) const;
  };
  // (end of struct MuonCollectionStoreManager)
 
  template<typename I>
  void 
  MuonCollectionStoreManager::cloneAndStore( const I & begin, const I & end, edm::Event & evt ) 
  {
      using namespace reco;
      rHits_ = evt.template getRefBeforePut<TrackingRecHitCollection>("TrackerOnly");
      rGBHits_ = evt.template getRefBeforePut<TrackingRecHitCollection>("GlobalMuon");
      rSAHits_ = evt.template getRefBeforePut<TrackingRecHitCollection>("StandAlone");
      rTrackExtras_ = evt.template getRefBeforePut<TrackExtraCollection>("TrackerOnly");
      rGBTrackExtras_ = evt.template getRefBeforePut<TrackExtraCollection>("GlobalMuon");
      rSATrackExtras_ = evt.template getRefBeforePut<TrackExtraCollection>("StandAlone");
      rTracks_ = evt.template getRefBeforePut<TrackCollection>("TrackerOnly");      
      rGBTracks_ = evt.template getRefBeforePut<TrackCollection>("GlobalMuon");      
      rSATracks_ = evt.template getRefBeforePut<TrackCollection>("StandAlone");      
      rMuons_ = evt.template getRefBeforePut<MuonCollection>("SelectedMuons");      
      //--- New: save clusters too
      edm::RefProd<edmNew::DetSetVector<SiStripCluster> > rStripClusters
	= evt.template getRefBeforePut<edmNew::DetSetVector<SiStripCluster> >();

      edm::RefProd<edmNew::DetSetVector<SiPixelCluster> >  rPixelClusters 
	= evt.template getRefBeforePut<edmNew::DetSetVector<SiPixelCluster> >();

      id_=0; igbd_=0; isad_=0; 
      idx_ = 0; igbdx_=0; isadx_=0; 
      hidx_=0; higbdx_=0; hisadx_=0;
      clusterStorer_.clear();

      for( I i = begin; i != end; ++ i ) {
	const Muon & mu = * * i;
          //--- Clone this track, and store references aside
          processMuon( mu );
      }
      //--- Clone the clusters and fixup refs
      clusterStorer_.processAllClusters(*selPixelClusters_, rPixelClusters,
					*selStripClusters_, rStripClusters);
   }
    
  //----------------------------------------------------------------------
  class MuonSelectorBase : public edm::EDFilter {
  public:
    MuonSelectorBase( const edm::ParameterSet & cfg ) {
      std::string alias( cfg.getParameter<std::string>( "@module_label" ) );


      produces<reco::MuonCollection>("SelectedMuons").setBranchAlias( alias + "SelectedMuons" );
      produces<reco::TrackCollection>("TrackerOnly").setBranchAlias( alias + "TrackerOnlyTracks" );
      produces<reco::TrackExtraCollection>("TrackerOnly").setBranchAlias( alias + "TrackerOnlyExtras" );
      produces<TrackingRecHitCollection>("TrackerOnly").setBranchAlias( alias + "TrackerOnlyHits" );
      //--- New: save clusters too
      produces< edmNew::DetSetVector<SiPixelCluster> >().setBranchAlias( alias + "PixelClusters" );
      produces< edmNew::DetSetVector<SiStripCluster> >().setBranchAlias( alias + "StripClusters" );
      produces<reco::TrackCollection>("GlobalMuon").setBranchAlias( alias + "GlobalMuonTracks" );
      produces<reco::TrackExtraCollection>("GlobalMuon").setBranchAlias( alias + "GlobalMuonExtras" );
      produces<TrackingRecHitCollection>("GlobalMuon").setBranchAlias( alias + "GlobalMuonHits" );
      produces<reco::TrackCollection>("StandAlone").setBranchAlias( alias + "StandAloneTracks" );
      produces<reco::TrackExtraCollection>("StandAlone").setBranchAlias( alias + "StandAloneExtras" );
      produces<TrackingRecHitCollection>("StandAlone").setBranchAlias( alias + "StandAloneHits" );

    }
   }; // (end of class MuonSelectorBase)


  template<>
  struct StoreManagerTrait<reco::MuonCollection> {
    typedef MuonCollectionStoreManager type;
    typedef MuonSelectorBase base;
  };

}

#endif
