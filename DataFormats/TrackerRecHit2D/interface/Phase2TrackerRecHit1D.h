#ifndef DATAFORMATS_TRACKERRECHIT2D_PHASE2TRACKERRECHIT1D_H 
#define DATAFORMATS_TRACKERRECHIT2D_PHASE2TRACKERRECHIT1D_H 

#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "DataFormats/TrackerRecHit2D/interface/TrackerSingleRecHit.h"

#include "TkCloner.h"


class Phase2TrackerRecHit1D final : public TrackerSingleRecHit {

public:

  typedef OmniClusterRef::Phase2Cluster1DRef CluRef;

  Phase2TrackerRecHit1D() {}

  ~Phase2TrackerRecHit1D() {}

  Phase2TrackerRecHit1D( const LocalPoint& pos, const LocalError& err, 
                         GeomDet const & idet,
		         CluRef const&  clus) : TrackerSingleRecHit(pos,err,idet,clus){}

  virtual Phase2TrackerRecHit1D * clone() const override { return new Phase2TrackerRecHit1D( * this); }
  virtual RecHitPointer cloneSH() const override { return std::make_shared<Phase2TrackerRecHit1D>(*this);}

  CluRef cluster()  const { return cluster_phase2OT(); }
  void setClusterRef(CluRef const & ref)  {setClusterPhase2Ref(ref);}

  virtual bool isPhase2() const override { return true; }
  //FIXME::check dimension of this!!
  virtual int dimension() const override {return 2;}
  virtual void getKfComponents( KfComponentsHolder & holder ) const override { getKfComponents2D(holder); }

  virtual bool canImproveWithTrack() const override {return true;}

private:
  // double dispatch
  virtual Phase2TrackerRecHit1D * clone(TkCloner const& cloner, TrajectoryStateOnSurface const& tsos) const override {
    return cloner(*this,tsos).release();
  }
  virtual  RecHitPointer cloneSH(TkCloner const& cloner, TrajectoryStateOnSurface const& tsos) const override {
    return cloner.makeShared(*this,tsos);
  }
};

typedef edmNew::DetSetVector< Phase2TrackerRecHit1D > Phase2TrackerRecHit1DCollectionNew;

#endif
