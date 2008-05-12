process Alignment =
{
  include "Alignment/HIPAlignmentAlgorithm/home<PATH>/../common.cff"
  include "RecoTracker/TransientTrackingRecHit/data/TransientTrackingRecHitBuilderWithoutRefit.cfi"
  include "RecoTracker/TrackProducer/data/RefitterWithMaterial.cff"
  include "RecoVertex/BeamSpotProducer/data/BeamSpot.cff"

  source = PoolSource { untracked vstring fileNames = {<FILE>} }

  untracked PSet maxEvents = { untracked int32 input = -1 }

# Patch for track refitter (adapted to alignment needs)

  replace TrackRefitter.src = <SKIM>
  replace TrackRefitter.TTRHBuilder = "WithoutRefit"
  replace TrackRefitter.TrajectoryInEvent = true
  replace ttrhbwor.Matcher = "StandardMatcher" # matching for strip stereo!

  replace HIPAlignmentAlgorithm.outpath = "<PATH>/"
  replace HIPAlignmentAlgorithm.verbosity = true
  replace HIPAlignmentAlgorithm.apeParam =
  {
    {
      PSet Selector = { vstring alignParams = {"PixelHalfBarrelDets,000000"} }
      string function = "linear" # linear or exponential
      vdouble apeSPar = {2e-2, 0.0, 100}
      vdouble apeRPar = {3e-4, 0.0, 100}
    },
    {
      PSet Selector = { vstring alignParams = {"PXECDets,000000"} }
      string function = "linear" # linear or exponential
      vdouble apeSPar = {5e-3, 0.0, 100}
      vdouble apeRPar = {1e-3, 0.0, 100}
    },
    {
      PSet Selector = { vstring alignParams = {"TIBDets,000000"} }
      string function = "linear" # linear or exponential
      vdouble apeSPar = {5e-2, 0.0, 100}
      vdouble apeRPar = {5e-4, 0.0, 100}
    },
    {
      PSet Selector = { vstring alignParams = {"TIDDets,000000"} }
      string function = "linear" # linear or exponential
      vdouble apeSPar = {4e-2, 0.0, 100}
      vdouble apeRPar = {1e-3, 0.0, 100}
    },
    {
      PSet Selector = { vstring alignParams = {"TOBDets,000000"} }
      string function = "linear" # linear or exponential
      vdouble apeSPar = {4e-2, 0.0, 100}
      vdouble apeRPar = {1e-4, 0.0, 100}
    },
    {
      PSet Selector = { vstring alignParams = {"TECDets,000000"} }
      string function = "linear" # linear or exponential
      vdouble apeSPar = {1e-2, 0.0, 100}
      vdouble apeRPar = {1e-4, 0.0, 100}
    }
  }

  path p = { offlineBeamSpot, TrackRefitter }
}
