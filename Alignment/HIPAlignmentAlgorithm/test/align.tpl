process Alignment =
{
  include "Alignment/CommonAlignmentProducer/data/AlignmentTrackSelector.cfi"
  include "../home<PATH>/../common.cff"
  include "../home<PATH>/../<SKIM>TrackSelection.cff"
  include "RecoTracker/TransientTrackingRecHit/data/TransientTrackingRecHitBuilderWithoutRefit.cfi"
  include "RecoTracker/TrackProducer/data/RefitterWithMaterial.cff"
  include "RecoVertex/BeamSpotProducer/data/BeamSpot.cff"

  source = PoolSource
  {
    untracked bool useCSA08Kludge = true
    untracked vstring fileNames = {<FILE>}
  }

# Track selections

  replace AlignmentTrackSelector.src = <SKIM>
 
# Patch for track refitter (adapted to alignment needs)

  replace TrackRefitter.src = AlignmentTrackSelector
  replace TrackRefitter.TTRHBuilder = "WithoutRefit"
  replace TrackRefitter.TrajectoryInEvent = true
  replace ttrhbwor.Matcher = "StandardMatcher" # matching for strip stereo!

  replace HIPAlignmentAlgorithm.outpath  = ""
  replace HIPAlignmentAlgorithm.uvarFile = "<PATH>/IOUserVariables.root"

  replace HIPAlignmentAlgorithm.apeParam =
  {
    {
      PSet Selector = { vstring alignParams = {"TrackerTPBModule,000000"} }
      string function = "linear" # linear or exponential
      vdouble apeSPar = {1e-2, 8e-3, 10}
      vdouble apeRPar = {3e-3, 3e-3, 10}
    },
    {
      PSet Selector = { vstring alignParams = {"TrackerTPEModule,000000"} }
      string function = "linear" # linear or exponential
      vdouble apeSPar = {1e-2, 1e-2, 10}
      vdouble apeRPar = {3e-3, 3e-3, 10}
    },
    {
      PSet Selector = { vstring alignParams = {"TIBDets,000000"} }
      string function = "linear" # linear or exponential
      vdouble apeSPar = {5e-2, 3e-2, 10}
      vdouble apeRPar = {2e-3, 2e-3, 10}
    },
    {
      PSet Selector = { vstring alignParams = {"TIDDets,000000"} }
      string function = "linear" # linear or exponential
      vdouble apeSPar = {6e-2, 5e-2, 10}
      vdouble apeRPar = {2e-3, 2e-3, 10}
    },
    {
      PSet Selector = { vstring alignParams = {"TOBDets,000000"} }
      string function = "linear" # linear or exponential
      vdouble apeSPar = {2e-2, 2e-2, 10}
      vdouble apeRPar = {6e-4, 6e-4, 10}
    },
    {
      PSet Selector = { vstring alignParams = {"TECDets,000000"} }
      string function = "linear" # linear or exponential
      vdouble apeSPar = {2e-2, 2e-2, 10}
      vdouble apeRPar = {7e-4, 7e-4, 10}
    }
  }

  path p = { offlineBeamSpot, AlignmentTrackSelector, TrackRefitter }
}
