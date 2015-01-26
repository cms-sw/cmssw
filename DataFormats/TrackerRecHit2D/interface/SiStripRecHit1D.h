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

 
  ClusterRef cluster()  const { return cluster_strip() ; }
  void setClusterRef(ClusterRef const & ref)  {setClusterStripRef(ref);}


  virtual SiStripRecHit1D * clone() const GCC11_OVERRIDE {return new SiStripRecHit1D( * this); }
#ifndef __GCCXML__
  virtual RecHitPointer cloneSH() const { return std::make_shared<SiStripRecHit1D>(*this);}
#endif
  

  virtual int dimension() const GCC11_OVERRIDE {return 1;}
  virtual void getKfComponents( KfComponentsHolder & holder ) const GCC11_OVERRIDE {getKfComponents1D(holder);}

  virtual bool canImproveWithTrack() const GCC11_OVERRIDE {return true;}
private:
  // double dispatch
  virtual SiStripRecHit1D * clone(TkCloner const& cloner, TrajectoryStateOnSurface const& tsos) const GCC11_OVERRIDE {
    return cloner(*this,tsos).release();
  }
#ifndef __GCCXML__
  virtual  RecHitPointer cloneSH(TkCloner const& cloner, TrajectoryStateOnSurface const& tsos) const GCC11_OVERRIDE {
    return cloner.makeShared(*this,tsos);
  }
#endif 

 
};

#endif
