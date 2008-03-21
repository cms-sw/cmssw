process Alignment =
{
  include "Alignment/HIPAlignmentAlgorithm/test/common.cff"

  source = PoolSource { untracked vstring fileNames = {<FILE>} }

  untracked PSet maxEvents = { untracked int32 input = -1 }

# Patch for track refitter (adapted to alignment needs)

  include "RecoTracker/TransientTrackingRecHit/data/TransientTrackingRecHitBuilderWithoutRefit.cfi"
  include "RecoTracker/TrackProducer/data/RefitterWithMaterial.cff"

  replace TrackRefitter.src = ALCARECOTkAlZMuMu
  replace TrackRefitter.TTRHBuilder = "WithoutRefit"
  replace TrackRefitter.TrajectoryInEvent = true
  replace ttrhbwor.Matcher = "StandardMatcher" # matching for strip stereo!

  replace HIPAlignmentAlgorithm.outpath = "<PATH>/"
  replace HIPAlignmentAlgorithm.apeSPar = {0.10, 0.05, 10}
  replace HIPAlignmentAlgorithm.apeRPar = {1e-3, 5e-4, 10}
  replace HIPAlignmentAlgorithm.minimumNumberOfHits = 0

  path p = { TrackRefitter }
}
