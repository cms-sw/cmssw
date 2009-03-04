#ifndef RecoAlgos_TrackSelector_h
#define RecoAlgos_TrackSelector_h
/** \class TrackSelector
 *
 * selects a subset of a track collection. Also clones
 * TrackExtra part and RecHits collection
 * 
 * \author Luca Lista, INFN
 *         Reorganized by Petar Maksimovic, JHU
 *         Reorganized again by Giovanni Petrucciani
 * \version $Revision: 1.23 $
 *
 * $Id: TrackSelector.h,v 1.23 2008/06/06 18:01:40 gpetrucc Exp $
 *
 */
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"


#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
// Apparently this is not anywhere defined
typedef edm::RefProd<SiStripClusterCollection> SiStripClusterRefProd;

namespace helper {

  //------------------------------------------------------------------
  //!  \brief Class to manage copying of RecHits and Clusters from Tracks.
  //------------------------------------------------------------------
  struct TrackCollectionStoreManager {
  public:
    typedef reco::TrackCollection collection;

    TrackCollectionStoreManager(const edm::Handle<reco::TrackCollection> & );

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
    edm::OrphanHandle<reco::TrackCollection> put( edm::Event & evt );

    //------------------------------------------------------------------
    //!  Get the size.
    //------------------------------------------------------------------
    inline size_t size() const { return selTracks_->size(); }
    

    //------------------------------------------------------------------
    //! \brief Method to clone tracks, track extras and their hits and clusters.
    //! typename I = this is an interator over a track collection, **I needs
    //! to dereference into a Track.
    //------------------------------------------------------------------
    template<typename I>
    void cloneAndStore( const I & begin, const I & end, edm::Event & evt ) ;
    
  private:
    //--- A struct for clusters associated to hits
    template<typename RecHitType, typename ClusterRefType = typename RecHitType::ClusterRef>
    class ClusterHitRecord {
        public:
            /// Create a record for a hit with a given index in the TrackingRecHitCollection
            ClusterHitRecord(const RecHitType &hit, size_t idx) : 
                detid_(hit.geographicalId().rawId()), index_(idx), ref_(hit.cluster()) {}
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
            void rekey(TrackingRecHitCollection &hits, const ClusterRefType &newRef) const ;
        private:
            uint32_t detid_;
            size_t   index_;
            ClusterRefType ref_;
    };
    typedef ClusterHitRecord<SiPixelRecHit>   PixelClusterHitRecord;
    typedef ClusterHitRecord<SiStripRecHit2D> StripClusterHitRecord;

    //--- Collections:
    std::auto_ptr<reco::TrackCollection>                   selTracks_;
    std::auto_ptr<reco::TrackExtraCollection>              selTrackExtras_;
    std::auto_ptr<TrackingRecHitCollection>                selHits_;
    std::auto_ptr< edmNew::DetSetVector<SiStripCluster> >  selStripClusters_;
    std::auto_ptr< edmNew::DetSetVector<SiPixelCluster> >  selPixelClusters_;
    //--- Information about the cloned clusters
    std::vector<StripClusterHitRecord>                     stripClusterRecords_;
    std::vector<PixelClusterHitRecord>                     pixelClusterRecords_;
    //--- RefProd<>'s
    reco::TrackRefProd           rTracks_ ;
    reco::TrackExtraRefProd      rTrackExtras_ ;
    TrackingRecHitRefProd        rHits_ ;
    edm::RefProd< edmNew::DetSetVector<SiStripCluster> > rStripClusters_ ;
    edm::RefProd< edmNew::DetSetVector<SiPixelCluster> > rPixelClusters_ ;

    //--- Indices into collections handled with RefProd
    size_t idx_   ;      //!<  index to track extra coll
    size_t hidx_  ;      //!<  index to tracking rec hits

    //--- Switches 
    bool   cloneClusters_ ;  //!< Clone clusters, or not?  Default: true.
    
    //--- Methods
    //------------------------------------------------------------------
    //!  Process a single track.  
    //------------------------------------------------------------------
    void processTrack( const reco::Track & trk );

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


  };
  // (end of struct TrackCollectionStoreManager)

  
  template<typename I>
  void 
  TrackCollectionStoreManager::cloneAndStore( const I & begin, const I & end, edm::Event & evt ) 
  {
      using namespace reco;

      rTracks_       = evt.template getRefBeforePut<TrackCollection>();      
      rTrackExtras_  = evt.template getRefBeforePut<TrackExtraCollection>();
      rHits_         = evt.template getRefBeforePut<TrackingRecHitCollection>();
      //--- New: save clusters too
      rStripClusters_ 
          = evt.template getRefBeforePut< edmNew::DetSetVector<SiStripCluster> >();

      rPixelClusters_ 
          = evt.template getRefBeforePut< edmNew::DetSetVector<SiPixelCluster> >();

      //--- Indices into collections handled with RefProd
      idx_ = 0;         //!<  index to track extra coll
      hidx_ = 0;        //!<  index to tracking rec hits

      //--- Records about the clusters we want to clone
      stripClusterRecords_.clear();      
      pixelClusterRecords_.clear();      

      //--- Loop over tracks
      for( I i = begin; i != end; ++ i ) {
          //--- Whatever type the iterator i is, deref to reco::Track &
          const reco::Track & trk = * * i;
          //--- Clone this track, and store references aside
          processTrack( trk );
      }
      //--- Clone the clusters and fixup refs
      processAllClusters();
  }


  //----------------------------------------------------------------------
  class TrackSelectorBase : public edm::EDFilter {
  public:
    TrackSelectorBase( const edm::ParameterSet & cfg ) {
      std::string alias( cfg.getParameter<std::string>( "@module_label" ) );
      produces<reco::TrackCollection>().setBranchAlias( alias + "Tracks" );
      produces<reco::TrackExtraCollection>().setBranchAlias( alias + "TrackExtras" );
      produces<TrackingRecHitCollection>().setBranchAlias( alias + "RecHits" );
      //--- New: save clusters too
      // FIXME: For the following two, need to check what names
      // FIXME: of the output collections are needed downstream.
      produces< edmNew::DetSetVector<SiPixelCluster> >().setBranchAlias( alias + "PixelClusters" );
      produces< edmNew::DetSetVector<SiStripCluster> >().setBranchAlias( alias + "StripClusters" );
    }
  };  // (end of class TrackSelectorBase)


  template<>
  struct StoreManagerTrait<reco::TrackCollection> {
    typedef TrackCollectionStoreManager type;
    typedef TrackSelectorBase base;
  };

}

#endif
