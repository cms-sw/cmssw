#ifndef SiStripRecHit2D_H
#define SiStripRecHit2D_H

#include "DataFormats/TrackerRecHit2D/interface/TrackerSingleRecHit.h"
#include "TkCloner.h"


class SiStripRecHit2D GCC11_FINAL : public TrackerSingleRecHit {
public:

  SiStripRecHit2D() {}

  ~SiStripRecHit2D() {} 

  typedef OmniClusterRef::ClusterStripRef         ClusterRef;

  // no position (as in persistent)
  SiStripRecHit2D(const DetId& id,
		  OmniClusterRef const& clus) : 
    TrackerSingleRecHit(id, clus){}

  template<typename CluRef>
  SiStripRecHit2D( const LocalPoint& pos, const LocalError& err,
		   GeomDet const & idet,
		   CluRef const& clus) : 
    TrackerSingleRecHit(pos,err, idet, clus) {}
 
				
  ClusterRef cluster()  const { return cluster_strip() ; }
  void setClusterRef(ClusterRef const & ref)  {setClusterStripRef(ref);}

  virtual SiStripRecHit2D * clone() const {return new SiStripRecHit2D( * this); }
  
  virtual int dimension() const {return 2;}
  virtual void getKfComponents( KfComponentsHolder & holder ) const { getKfComponents2D(holder); }

  virtual bool canImproveWithTrack() const {return true;}
private:
  // double dispatch
  virtual SiStripRecHit2D * clone(TkCloner const& cloner, TrajectoryStateOnSurface const& tsos) const {
    return cloner(*this,tsos);
  }
 
  
private:
 
};

#endif
