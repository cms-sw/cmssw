#ifndef L1TMUONBARRELKALMANTRACKFINDER_H
#define L1TMUONBARRELKALMANTRACKFINDER_H

#include "DataFormats/L1TMuon/interface/L1MuKBMTCombinedStub.h"
#include "L1Trigger/L1TMuonBarrel/interface/L1TMuonBarrelKalmanAlgo.h"
#include "L1Trigger/L1TMuonBarrel/interface/L1TMuonBarrelKalmanSectorProcessor.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class L1TMuonBarrelKalmanTrackFinder {
public:
  L1TMuonBarrelKalmanTrackFinder(const edm::ParameterSet&);
  ~L1TMuonBarrelKalmanTrackFinder();

  L1MuKBMTrackCollection process(L1TMuonBarrelKalmanAlgo*, const L1MuKBMTCombinedStubRefVector& stubs, int bx);

private:
  int verbose_;
  std::vector<L1TMuonBarrelKalmanSectorProcessor> sectors_;
};

#endif
