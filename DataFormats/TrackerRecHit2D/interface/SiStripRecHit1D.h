#ifndef SiStripRecHit1D_H
#define SiStripRecHit1D_H



#include "DataFormats/TrackerRecHit2D/interface/TrackerSingleRecHit.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"

#include "TkCloner.h"


class SiStripRecHit1D GCC11_FINAL : public TrackerSingleRecHit { 
public:

 
  SiStripRecHit1D(){}
  
  
  typedef OmniClusterRef::ClusterStripRef         ClusterRef;

  template<typename CluRef>
  SiStripRecHit1D( const LocalPoint& p, const LocalError& e,
		   GeomDet const & idet,
		   CluRef const&  clus) : TrackerSingleRecHit(p,e,idet,clus){}

 
  /// method to facilitate the convesion from 2D to 1D hits
  SiStripRecHit1D(const SiStripRecHit2D*);

  ClusterRef cluster()  const { return cluster_strip() ; }
  void setClusterRef(ClusterRef const & ref)  {setClusterStripRef(ref);}


  virtual SiStripRecHit1D * clone() const {return new SiStripRecHit1D( * this); }
  

  virtual int dimension() const {return 1;}
  virtual void getKfComponents( KfComponentsHolder & holder ) const {getKfComponents1D(holder);}

  virtual bool canImproveWithTrack() const {return true;}
private:
  // double dispatch
  virtual SiStripRecHit1D * clone(TkCloner const& cloner, TrajectoryStateOnSurface const& tsos) const {
    return cloner(*this,tsos);
  }
 

 
};

#endif
