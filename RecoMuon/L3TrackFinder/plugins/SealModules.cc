#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMuon/L3TrackFinder/interface/MuonCkfTrajectoryBuilder.h"
#include "RecoTracker/CkfPattern/interface/BaseCkfTrajectoryBuilderFactory.h"

DEFINE_EDM_PLUGIN(BaseCkfTrajectoryBuilderFactory, MuonCkfTrajectoryBuilder, "MuonCkfTrajectoryBuilder");
