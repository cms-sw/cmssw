#ifndef RecoAlgos_ClusterStorer_h
#define RecoAlgos_ClusterStorer_h
/** \class ClusterStorer
 *
 * Helper to store clones of SiStrip- and SiPixelClusters
 * of selected RecHits
 * 
 * \author Gero Flucke, DESY
 *         based on code originally part of TrackSelector.h by Giovanni Petrucciani,
 *         but extended to deal with both SiStripRecHit1D and SiStripRecHit2D
 * 
 * \version $Revision: 1.1 $
 *
 * $Id: ClusterStorer.h,v 1.1 2009/12/09 15:58:42 flucke Exp $
 *
 */
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

namespace helper {

  class ClusterStorer {
  public:
    ClusterStorer () {}
    /// add cluster of newHit to list (throws if hit is of unknown type)
    void addCluster(TrackingRecHitCollection &hits, size_t index);
    /// clear records
    void clear();
    //------------------------------------------------------------------
    //!  Processes all the clusters of the tracks 
    //!  (after the tracks have been dealt with),
    //!  need Refs to products (i.e. full collections) in the event.
    //------------------------------------------------------------------
    void processAllClusters(edmNew::DetSetVector<SiPixelCluster> &pixelDsvToFill,		    
			    edm::RefProd<edmNew::DetSetVector<SiPixelCluster> > refPixelClusters,
			    edmNew::DetSetVector<SiStripCluster> &stripDsvToFill,
			    edm::RefProd<edmNew::DetSetVector<SiStripCluster> > refStripClusters);
   
    private:
    /// A struct for clusters associated to hits
    template<typename ClusterRefType>
    class ClusterHitRecord {
    public:
      /// Create a record for a hit with a given index in the TrackingRecHitCollection.
      /// 'RecHitType' must have a method 'cluster()' that returns a 'ClusterRefType'.
      template<typename RecHitType>
      ClusterHitRecord(const RecHitType &hit, TrackingRecHitCollection &hits, size_t idx)
        : detid_(hit.geographicalId().rawId()), hits_(&hits), index_(idx), ref_(hit.cluster()) {}
      /// returns the detid
      uint32_t detid() const { return detid_; }
      /// this method is to be able to compare and see if two refs are the same
      const ClusterRefType & clusterRef() const { return ref_; }
      /// this one is to sort by detid and then by index of the rechit
      bool operator<(const ClusterHitRecord<ClusterRefType> &other) const
      {
        return (detid_ != other.detid_) ? detid_ < other.detid_ : ref_  < other.ref_;
      }
      /// Set the reference of the hit of this record to 'newRef',
      /// will not modify the ref stored in this object.
      template <typename RecHitType>
      void rekey(const ClusterRefType &newRef) const;
    private:
      ClusterHitRecord() {}/// private => unusable
      uint32_t detid_;
      TrackingRecHitCollection *hits_;
      size_t   index_;
      ClusterRefType ref_;
    };
    
    typedef ClusterHitRecord<SiPixelRecHit::ClusterRef>   PixelClusterHitRecord;
    /// Assuming that the ClusterRef is the same for all SiStripRecHit*:
    typedef ClusterHitRecord<SiStripRecHit2D::ClusterRef> StripClusterHitRecord;

    //------------------------------------------------------------------
    //!  Processes all the clusters of a specific type
    //!  (after the tracks have been dealt with)
    //------------------------------------------------------------------
    template<typename HitType, typename ClusterType>
      void
      processClusters(std::vector<ClusterHitRecord<typename HitType::ClusterRef> > &clusterRecords,
		      edmNew::DetSetVector<ClusterType>                            &dsvToFill,
		      edm::RefProd< edmNew::DetSetVector<ClusterType> >            &refprod);

    //--- Information about the cloned clusters
    std::vector<PixelClusterHitRecord>                  pixelClusterRecords_;
    std::vector<StripClusterHitRecord>                  stripClusterRecords_;
  };

}

#endif
