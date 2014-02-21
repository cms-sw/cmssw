#ifndef SiStripRecHit1D_H
#define SiStripRecHit1D_H



#include "DataFormats/TrackerRecHit2D/interface/TrackerSingleRecHit.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"

class SiStripRecHit1D GCC11_FINAL : public TrackerSingleRecHit { 
public:

 
  SiStripRecHit1D(): sigmaPitch_(-1.){}
  
  
  typedef OmniClusterRef::ClusterStripRef         ClusterRef;
  typedef OmniClusterRef::ClusterRegionalRef ClusterRegionalRef;

  template<typename CluRef>
  SiStripRecHit1D( const LocalPoint& p, const LocalError& e,
		   const DetId& id, GeomDet const * idet,
		   CluRef const&  clus) : TrackerSingleRecHit(p,e,id,idet,clus), sigmaPitch_(-1.){}

 
  /// method to facilitate the convesion from 2D to 1D hits
  SiStripRecHit1D(const SiStripRecHit2D*);

  ClusterRef cluster()  const { return cluster_strip() ; }
  void setClusterRef(ClusterRef const & ref)  {setClusterStripRef(ref);}


  virtual SiStripRecHit1D * clone() const {return new SiStripRecHit1D( * this); }
  

  virtual int dimension() const {return 1;}
  virtual void getKfComponents( KfComponentsHolder & holder ) const {getKfComponents1D(holder);}

 
  double sigmaPitch() const { return sigmaPitch_;}
  void setSigmaPitch(double sigmap) const { sigmaPitch_=sigmap;}

private:
 
 /// cache for the matcher....
  mutable double sigmaPitch_;  // transient.... 
};

#endif
