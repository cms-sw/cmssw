#ifndef L1TMUONBARRELKALMANREGIONMODULE_H
#define L1TMUONBARRELKALMANREGIONMODULE_H

#include "DataFormats/L1TMuon/interface/L1MuKBMTCombinedStub.h"
#include "L1Trigger/L1TMuonBarrel/interface/L1TMuonBarrelKalmanAlgo.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class L1TMuonBarrelKalmanRegionModule {
 public:
  L1TMuonBarrelKalmanRegionModule(const edm::ParameterSet&,int wheel,int sector);
  ~L1TMuonBarrelKalmanRegionModule();


  L1MuKBMTrackCollection process(L1TMuonBarrelKalmanAlgo*,const L1MuKBMTCombinedStubRefVector& stubs,int bx);
 private:
  int verbose_;
  int sector_;
  int wheel_;
  int nextSector_;
  int previousSector_;
  int nextWheel_;

  L1MuKBMTrackCollection cleanRegion(const L1MuKBMTrackCollection&,const L1MuKBMTrackCollection&,const L1MuKBMTrackCollection&);
  L1MuKBMTrackCollection selfClean(const L1MuKBMTrackCollection& tracks);
  L1MuKBMTrackCollection cleanHigher(const L1MuKBMTrackCollection& tracks1,const L1MuKBMTrackCollection& tracks2);
  L1MuKBMTrackCollection cleanLower(const L1MuKBMTrackCollection& tracks1,const L1MuKBMTrackCollection& tracks2);

  class TrackSorter {
  public:
    TrackSorter() {
    }

    bool operator() (const L1MuKBMTrack& a ,const L1MuKBMTrack& b) {
      if (abs(a.curvatureAtVertex())<=abs(b.curvatureAtVertex()))
	return true;
      return false;
    }
  };
  



  
};



#endif
