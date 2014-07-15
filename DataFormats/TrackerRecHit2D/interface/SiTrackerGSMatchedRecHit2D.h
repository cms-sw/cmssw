#ifndef SiTrackerGSMatchedRecHit2D_H
#define SiTrackerGSMatchedRecHit2D_H

#include "DataFormats/TrackerRecHit2D/interface/GSSiTrackerRecHit2DLocalPos.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"
class SiTrackerGSRecHit2D;

class SiTrackerGSMatchedRecHit2D : public GSSiTrackerRecHit2DLocalPos{
  
public:
  
  SiTrackerGSMatchedRecHit2D(): GSSiTrackerRecHit2DLocalPos(),
			 simhitId_(),
			 simtrackId_(),
			 eeId_(),
                         cluster_(),  
                         pixelMultiplicityAlpha_(), 
                         pixelMultiplicityBeta_(),
                         isMatched_(), 
                         componentMono_(),
                         componentStereo_() {}
  
  ~SiTrackerGSMatchedRecHit2D() {}
  
 typedef edm::Ref<FastTrackerClusterCollection, FastTrackerCluster > ClusterRef;
 typedef edm::RefProd<FastTrackerClusterCollection> ClusterRefProd;


  SiTrackerGSMatchedRecHit2D( const LocalPoint&, const LocalError&,
		       GeomDet const & idet,
		       const int simhitId,
		       const int simtrackId,
		       const uint32_t eeId,
                       ClusterRef const&  cluster,
		       const int pixelMultiplicityX,
		       const int pixelMultiplicityY,
		       const bool isMatched,
		       const SiTrackerGSRecHit2D* rMono, 
		       const SiTrackerGSRecHit2D* rStereo 
		       );  

  SiTrackerGSMatchedRecHit2D( const LocalPoint&, const LocalError&,
		       GeomDet const & idet,
		       const int simhitId,
		       const int simtrackId,
		       const uint32_t eeId,
                       ClusterRef const&  cluster,
		       const int pixelMultiplicityX,
		       const int pixelMultiplicityY
		       );  

  virtual SiTrackerGSMatchedRecHit2D * clone() const {return new SiTrackerGSMatchedRecHit2D( * this); }
  
  const int& simhitId()    const { return simhitId_;}
  const int& simtrackId()  const { return simtrackId_;}
  const uint32_t& eeId()   const { return eeId_;}
  const int& simMultX()    const { return pixelMultiplicityAlpha_;}
  const int& simMultY()    const { return pixelMultiplicityBeta_;}
  const bool& isMatched()  const { return isMatched_;}
  const SiTrackerGSRecHit2D *monoHit() const { return &componentMono_;}
  const SiTrackerGSRecHit2D *stereoHit() const { return &componentStereo_;}

 ClusterRef const& cluster() const { return cluster_;}
  void setClusterRef(const ClusterRef &ref) { cluster_  = ref; }
 

  virtual bool sharesInput( const TrackingRecHit* other, SharedInputType what) const;
 
private:
  int const simhitId_;
  int const simtrackId_;
  uint32_t const eeId_;
  ClusterRef cluster_;
  int const pixelMultiplicityAlpha_;
  int const pixelMultiplicityBeta_;
  bool isMatched_;

  SiTrackerGSRecHit2D componentMono_;
  SiTrackerGSRecHit2D componentStereo_;
};



#endif
