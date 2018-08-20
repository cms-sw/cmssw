#ifndef SiStripRecHit1D_H
#define SiStripRecHit1D_H



#include "DataFormats/TrackerRecHit2D/interface/TrackerSingleRecHit.h"

#include "TkCloner.h"


class SiStripRecHit1D final : public TrackerSingleRecHit { 
public:

 
  SiStripRecHit1D(){}
  
  
  typedef OmniClusterRef::ClusterStripRef         ClusterRef;

  template<typename CluRef>
  SiStripRecHit1D( const LocalPoint& p, const LocalError& e,
		   GeomDet const & idet,
		   CluRef const&  clus) : TrackerSingleRecHit(p,e,idet,clus){}

 
  ClusterRef cluster()  const { return cluster_strip() ; }
  void setClusterRef(ClusterRef const & ref)  {setClusterStripRef(ref);}


  SiStripRecHit1D * clone() const override {return new SiStripRecHit1D( * this); }
#ifndef __GCCXML__
  RecHitPointer cloneSH() const override { return std::make_shared<SiStripRecHit1D>(*this);}
#endif
  

  int dimension() const override {return 1;}
  void getKfComponents( KfComponentsHolder & holder ) const override {getKfComponents1D(holder);}

  bool canImproveWithTrack() const override {return true;}
private:
  // double dispatch
  SiStripRecHit1D * clone_(TkCloner const& cloner, TrajectoryStateOnSurface const& tsos) const override {
    return cloner(*this,tsos).release();
  }
#ifndef __GCCXML__
   RecHitPointer cloneSH_(TkCloner const& cloner, TrajectoryStateOnSurface const& tsos) const override {
    return cloner.makeShared(*this,tsos);
  }
#endif 

 
};

#endif
