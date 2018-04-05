#ifndef L1TMUONBARRELKALMANSECTORPROCESSOR_H
#define L1TMUONBARRELKALMANSECTORPROCESSOR_H

#include "DataFormats/L1TMuon/interface/L1MuKBMTCombinedStub.h"
#include "L1Trigger/L1TMuonBarrel/interface/L1TMuonBarrelKalmanAlgo.h"
#include "L1Trigger/L1TMuonBarrel/interface/L1TMuonBarrelKalmanRegionModule.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class L1TMuonBarrelKalmanSectorProcessor {
 public:
  L1TMuonBarrelKalmanSectorProcessor(const edm::ParameterSet&,int sector);
  ~L1TMuonBarrelKalmanSectorProcessor();

  L1MuKBMTrackCollection process(L1TMuonBarrelKalmanAlgo*,const L1MuKBMTCombinedStubRefVector& stubs,int bx);
 private:
  int verbose_;
  int sector_;

  std::vector<L1TMuonBarrelKalmanRegionModule> regions_;
};



#endif
