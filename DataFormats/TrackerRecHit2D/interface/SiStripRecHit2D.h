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

  virtual SiStripRecHit2D * clone() const GCC11_OVERRIDE {return new SiStripRecHit2D( * this); }
#ifndef __GCCXML__
  virtual RecHitPointer cloneSH() const GCC11_OVERRIDE { return std::make_shared<SiStripRecHit2D>(*this);}
#endif
  
  virtual int dimension() const GCC11_OVERRIDE {return 2;}
  virtual void getKfComponents( KfComponentsHolder & holder ) const GCC11_OVERRIDE { getKfComponents2D(holder); }

  virtual bool canImproveWithTrack() const GCC11_OVERRIDE {return true;}
private:
  // double dispatch
  virtual SiStripRecHit2D* clone(TkCloner const& cloner, TrajectoryStateOnSurface const& tsos) const GCC11_OVERRIDE {
    return cloner(*this,tsos).release();
  }
#ifndef __GCCXML__
  virtual  RecHitPointer cloneSH(TkCloner const& cloner, TrajectoryStateOnSurface const& tsos) const GCC11_OVERRIDE {
    return cloner.makeShared(*this,tsos);
  }
#endif 
  
private:
 
};

#endif
