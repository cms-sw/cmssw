#ifndef DATAFORMATS_TRACKERRECHIT2D_PHASE2TRACKERRECHIT1D_H 
#define DATAFORMATS_TRACKERRECHIT2D_PHASE2TRACKERRECHIT1D_H 

#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "DataFormats/TrackerRecHit2D/interface/TrackerSingleRecHit.h"

class Phase2TrackerRecHit1D final : public TrackerSingleRecHit {

public:

  typedef OmniClusterRef::Phase2Cluster1DRef CluRef;

  Phase2TrackerRecHit1D() {}

  ~Phase2TrackerRecHit1D() {}

  Phase2TrackerRecHit1D( const LocalPoint& pos, const LocalError& err, 
                         GeomDet const & idet,
		         CluRef const&  clus) : TrackerSingleRecHit(pos,err,idet,clus){}

  virtual Phase2TrackerRecHit1D * clone() const override { return new Phase2TrackerRecHit1D( * this); }
#ifndef __GCCXML__
  virtual RecHitPointer cloneSH() const override { return std::make_shared<Phase2TrackerRecHit1D>(*this);}
#endif

  CluRef cluster()  const { return cluster_phase2OT(); }
  void setClusterRef(CluRef const & ref)  {setClusterPhase2Ref(ref);}

  virtual bool isPhase2() const override { return true; }
  //FIXME::check dimension of this!!
  virtual int dimension() const override {return 2;}
  virtual void getKfComponents( KfComponentsHolder & holder ) const override { getKfComponents2D(holder); }

private:

};

typedef edmNew::DetSetVector< Phase2TrackerRecHit1D > Phase2TrackerRecHit1DCollectionNew;

#endif
