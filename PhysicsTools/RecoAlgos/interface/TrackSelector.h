#ifndef RecoAlgos_TrackSelector_h
#define RecoAlgos_TrackSelector_h
/** \class TrackSelector
 *
 * selects a subset of a track collection. Also clones
 * TrackExtra part and RecHits collection
 * 
 * \author Luca Lista, INFN
 *         Reorganized by Petar Maksimovic, JHU
 * \version $Revision: 1.20 $
 *
 * $Id: TrackSelector.h,v 1.20 2008/06/05 16:54:03 petar Exp $
 *
 */
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

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
    //!  Process a single track.  THIS IS WHERE ALL THE ACTION HAPPENS!
    //------------------------------------------------------------------
    void processTrack( const reco::Track & trk );

    //------------------------------------------------------------------
    //! \brief Method to clone tracks, track extras and their hits and clusters.
    //! typename I = this is an interator over a track collection, **I needs
    //! to dereference into a Track.
    //------------------------------------------------------------------
    template<typename I>
    void cloneAndStore( const I & begin, const I & end, edm::Event & evt ) {
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
      scidx_ = 0;       //!<  index to strip cluster coll
      pcidx_ = 0;       //!<  index to pixel cluster coll
      
      //--- Loop over tracks
      for( I i = begin; i != end; ++ i ) {
	//--- Whatever type the iterator i is, deref to reco::Track &
	const reco::Track & trk = * * i;
	//--- Clone this track & fix refs.  (This is where all the work is done.)
	processTrack( trk );
      }
    }
    
    
    

  private:
    //--- Collections:
    std::auto_ptr<reco::TrackCollection>                   selTracks_;
    std::auto_ptr<reco::TrackExtraCollection>              selTrackExtras_;
    std::auto_ptr<TrackingRecHitCollection>                selHits_;
    std::auto_ptr< edmNew::DetSetVector<SiStripCluster> >  selStripClusters_;
    std::auto_ptr< edmNew::DetSetVector<SiPixelCluster> >  selPixelClusters_;

    //--- RefProd<>'s
    reco::TrackRefProd           rTracks_ ;
    reco::TrackExtraRefProd      rTrackExtras_ ;
    TrackingRecHitRefProd        rHits_ ;
    edm::RefProd< edmNew::DetSetVector<SiStripCluster> > rStripClusters_ ;
    edm::RefProd< edmNew::DetSetVector<SiPixelCluster> > rPixelClusters_ ;

    //--- Indices into collections handled with RefProd
    size_t idx_   ;      //!<  index to track extra coll
    size_t hidx_  ;      //!<  index to tracking rec hits
    size_t scidx_ ;      //!<  index to strip cluster coll
    size_t pcidx_ ;      //!<  index to pixel cluster coll

    //--- Switches 
    bool   cloneClusters_ ;  //!< Clone clusters, or not?  Default: true.
  };
  // (end of struct TrackCollectionStoreManager)



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
