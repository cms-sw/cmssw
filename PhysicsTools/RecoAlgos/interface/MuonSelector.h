#ifndef RecoAlgos_MuonSelector_h
#define RecoAlgos_MuonSelector_h
/** \class MuonSelector
 *
 * selects a subset of a muon collection and clones
 * Track, TrackExtra parts and RecHits collection
 * for SA, GB and Tracker Only options
 * 
 * \author Javier Fernandez, Uniovi
 *
 * \version $Revision: 1.10 $
 *
 * $Id: MuonSelector.h,v 1.10 2008/09/16 08:56:24 jfernan2 Exp $
 *
 */
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"

#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
// Apparently this is not anywhere defined
typedef edm::RefProd<SiStripClusterCollection> SiStripClusterRefProd;

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
    //--- A struct for clusters associated to hits
    template<typename RecHitType, typename ClusterRefType = typename RecHitType::ClusterRef>
    class ClusterHitRecord {
        public:
            /// Create a record for a hit with a given index in the TrackingRecHitCollection
            ClusterHitRecord(const RecHitType &hit, edm::OwnVector<TrackingRecHit> * hitVector, size_t idx) : 
                detid_(hit.geographicalId().rawId()), hitVector_(hitVector), index_(idx), ref_(hit.cluster()) {}
            /// returns the detid
            uint32_t detid() const { return detid_; }
            /// this method is to be able to compare and see if two refs are the same
            const ClusterRefType & clusterRef() const { return ref_; }
            /// this one is to sort by detid and then by index of the rechit
            bool operator<(const ClusterHitRecord<RecHitType,ClusterRefType> &other) const { 
                return (detid_ != other.detid_) ? detid_ < other.detid_ : ref_  < other.ref_;
            }
            /// correct the corresponding hit in the TrackingRecHitCollection with the new cluster ref
            /// will not modify the ref stored in this object
            void rekey(const ClusterRefType &newRef) const ;
        private:
            uint32_t detid_;
            edm::OwnVector<TrackingRecHit> * hitVector_;
            size_t   index_;
            ClusterRefType ref_;
    };

    typedef ClusterHitRecord<SiPixelRecHit>   PixelClusterHitRecord;
    typedef ClusterHitRecord<SiStripRecHit2D> StripClusterHitRecord;

     //--- Collections:
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
    //--- Information about the cloned clusters
    std::vector<StripClusterHitRecord>                     stripClusterRecords_;
    std::vector<PixelClusterHitRecord>                     pixelClusterRecords_;

      reco::MuonRefProd rMuons_;      
 
      reco::TrackRefProd rTracks_;      
      reco::TrackExtraRefProd rTrackExtras_;
      TrackingRecHitRefProd rHits_;

	// New: clusters
    edm::RefProd< edmNew::DetSetVector<SiStripCluster> > rStripClusters_ ;
    edm::RefProd< edmNew::DetSetVector<SiPixelCluster> > rPixelClusters_ ;

      reco::TrackRefProd rGBTracks_;      
      reco::TrackExtraRefProd rGBTrackExtras_;
      TrackingRecHitRefProd rGBHits_;

      reco::TrackRefProd rSATracks_;      
      reco::TrackExtraRefProd rSATrackExtras_;
      TrackingRecHitRefProd rSAHits_;

    //--- Indices into collections handled with RefProd
    size_t id_, igbd_, isad_, idx_, igbdx_, isadx_, hidx_, higbdx_, hisadx_;

    //--- Switches 
    bool   cloneClusters_ ;  //!< Clone clusters, or not?  Default: true.
    
    //--- Methods
    //------------------------------------------------------------------
    //!  Process a single muon.  
    //------------------------------------------------------------------
    void processMuon( const reco::Muon & mu );

    //------------------------------------------------------------------
    //!  Process a single hit.  
    //------------------------------------------------------------------
    void processHit( const TrackingRecHit * hit, edm::OwnVector<TrackingRecHit> &hits );


    //------------------------------------------------------------------
    //!  Processes all the clusters of the tracks 
    //!  (after the tracks have been dealt with)
    //------------------------------------------------------------------
    void processAllClusters() ;
 
    //------------------------------------------------------------------
    //!  Processes all the clusters of a specific type
    //!  (after the tracks have been dealt with)
    //------------------------------------------------------------------
    template<typename HitType, typename ClusterType>
    void processClusters( std::vector<ClusterHitRecord<HitType> > & clusterRecords,
              edmNew::DetSetVector<ClusterType>                   & dsv,
              edm::RefProd< edmNew::DetSetVector<ClusterType> >   & refprod ) ;

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
      rStripClusters_ 
          = evt.template getRefBeforePut< edmNew::DetSetVector<SiStripCluster> >();

      rPixelClusters_ 
          = evt.template getRefBeforePut< edmNew::DetSetVector<SiPixelCluster> >();

      id_=0; igbd_=0; isad_=0; 
      idx_ = 0; igbdx_=0; isadx_=0; 
      hidx_=0; higbdx_=0; hisadx_=0;

      //--- Records about the clusters we want to clone
      stripClusterRecords_.clear();      
      pixelClusterRecords_.clear();      

      for( I i = begin; i != end; ++ i ) {
	const Muon & mu = * * i;
          //--- Clone this track, and store references aside
          processMuon( mu );
      }
      //--- Clone the clusters and fixup refs
      processAllClusters();
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
      // FIXME: For the following two, need to check what names
      // FIXME: of the output collections are needed downstream.
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
