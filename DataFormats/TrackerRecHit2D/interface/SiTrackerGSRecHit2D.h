#ifndef SiTrackerGSRecHit2D_H
#define SiTrackerGSRecHit2D_H

#include "DataFormats/TrackerRecHit2D/interface/GSSiTrackerRecHit2DLocalPos.h"
#include "DataFormats/Common/interface/Ref.h"
#include "FastSimDataFormats/External/interface/FastTrackerClusterCollection.h" 

// typedef edm::Ref<FastTrackerClusterCollection, FastTrackerCluster > ClusterRef;
// typedef edm::RefProd<FastTrackerClusterCollection> ClusterRefProd;

class SiTrackerGSRecHit2D : public GSSiTrackerRecHit2DLocalPos{
  
public:
  
 
  
  SiTrackerGSRecHit2D(): GSSiTrackerRecHit2DLocalPos(),
			 simhitId_(),
			 simtrackId_(),
			 eeId_(),
                         cluster_(),  
			 pixelMultiplicityAlpha_(), 
                         pixelMultiplicityBeta_() {}
  
  ~SiTrackerGSRecHit2D() {}
  
 typedef edm::Ref<FastTrackerClusterCollection, FastTrackerCluster > ClusterRef;
 typedef edm::RefProd<FastTrackerClusterCollection> ClusterRefProd;


  SiTrackerGSRecHit2D( const LocalPoint&, const LocalError&,
		       GeomDet const & idet,
		       const int simhitId,
		       const int simtrackId,
		       const uint32_t eeId,
		       ClusterRef const&  cluster,
		       const int pixelMultiplicityX,
		       const int pixelMultiplicityY);     
  
  virtual SiTrackerGSRecHit2D * clone() const {return new SiTrackerGSRecHit2D( * this); }
  
  const int& simhitId()    const { return simhitId_;}
  const int& simtrackId()  const { return simtrackId_;}
  const uint32_t& eeId()   const { return eeId_;}
  const int& simMultX()    const { return pixelMultiplicityAlpha_;}
  const int& simMultY()    const { return pixelMultiplicityBeta_;}

  ClusterRef const& cluster() const { return cluster_;}
  void setClusterRef(const ClusterRef &ref) { cluster_  = ref; }
  
  virtual bool sharesInput( const TrackingRecHit* other, SharedInputType what) const {return false;}
  
 private:
  
  int simhitId_;
  int simtrackId_;
  uint32_t eeId_;
  ClusterRef cluster_;
  int pixelMultiplicityAlpha_;
  int pixelMultiplicityBeta_;
  
};

// Comparison operators
inline bool operator<( const SiTrackerGSRecHit2D& one, const SiTrackerGSRecHit2D& other) {
  if ( one.geographicalId() < other.geographicalId() ) {
    return true;
  } else {
    return false;
  }
}

#endif
