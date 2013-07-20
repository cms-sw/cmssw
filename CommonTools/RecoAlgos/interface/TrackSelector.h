#ifndef RecoAlgos_TrackSelector_h
#define RecoAlgos_TrackSelector_h
/** \class TrackSelector
 *
 * Selects a subset of a track collection. Also clones
 * TrackExtra part, RecHits and used SiStrip-/SiPixelCluster
 * 
 * \author Luca Lista, INFN
 *         Reorganized by Petar Maksimovic, JHU
 *         Reorganized again by Giovanni Petrucciani
 *         Outsourcing of cluster cloning by Gero Flucke, DESY
 * \version $Revision: 1.2 $
 *
 * $Id: TrackSelector.h,v 1.2 2009/12/09 16:00:33 flucke Exp $
 *
 */
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "CommonTools/RecoAlgos/interface/ClusterStorer.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"


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
    //--- Collections to store:
    std::auto_ptr<reco::TrackCollection>                   selTracks_;
    std::auto_ptr<reco::TrackExtraCollection>              selTrackExtras_;
    std::auto_ptr<TrackingRecHitCollection>                selHits_;
    std::auto_ptr< edmNew::DetSetVector<SiStripCluster> >  selStripClusters_;
    std::auto_ptr< edmNew::DetSetVector<SiPixelCluster> >  selPixelClusters_;

    //--- References to products (i.e. to collections):
    reco::TrackRefProd           rTracks_ ;
    reco::TrackExtraRefProd      rTrackExtras_ ;
    TrackingRecHitRefProd        rHits_ ;
    /// Helper to treat copies of selected clusters
    ///  and make the hits refer to the output cluster collections:
    ClusterStorer clusterStorer_;

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
      edm::RefProd<edmNew::DetSetVector<SiPixelCluster> > rPixelClusters
	= evt.template getRefBeforePut<edmNew::DetSetVector<SiPixelCluster> >();
      edm::RefProd<edmNew::DetSetVector<SiStripCluster> > rStripClusters 
	= evt.template getRefBeforePut<edmNew::DetSetVector<SiStripCluster> >();

      //--- Indices into collections handled with RefProd
      idx_ = 0;         //!<  index to track extra coll
      hidx_ = 0;        //!<  index to tracking rec hits
      clusterStorer_.clear();

      //--- Loop over tracks
      for( I i = begin; i != end; ++ i ) {
          //--- Whatever type the iterator i is, deref to reco::Track &
          const reco::Track & trk = * * i;
          //--- Clone this track, and store references aside
          processTrack( trk );
      }
      //--- Clone the clusters and fixup refs
      clusterStorer_.processAllClusters(*selPixelClusters_, rPixelClusters,
					*selStripClusters_, rStripClusters);
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
