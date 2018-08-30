#ifndef SiStripRecHit2D_H
#define SiStripRecHit2D_H

#include "DataFormats/TrackerRecHit2D/interface/TrackerSingleRecHit.h"
#include "TkCloner.h"


class SiStripRecHit2D final : public TrackerSingleRecHit {
public:

  SiStripRecHit2D() {}

  ~SiStripRecHit2D() override {} 

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

  SiStripRecHit2D * clone() const override {return new SiStripRecHit2D( * this); }
#ifndef __GCCXML__
  RecHitPointer cloneSH() const override { return std::make_shared<SiStripRecHit2D>(*this);}
#endif
  
  int dimension() const override {return 2;}
  void getKfComponents( KfComponentsHolder & holder ) const override { getKfComponents2D(holder); }

  bool canImproveWithTrack() const override {return true;}
private:
  // double dispatch
  SiStripRecHit2D* clone_(TkCloner const& cloner, TrajectoryStateOnSurface const& tsos) const override {
    return cloner(*this,tsos).release();
  }
#ifndef __GCCXML__
   RecHitPointer cloneSH_(TkCloner const& cloner, TrajectoryStateOnSurface const& tsos) const override {
    return cloner.makeShared(*this,tsos);
  }
#endif 
  
private:
 
};

#endif
