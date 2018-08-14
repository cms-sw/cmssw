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

  
};



#endif
