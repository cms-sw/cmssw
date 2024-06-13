import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFTracking.modules import ElectronSeedMerger
electronMergedSeeds = ElectronSeedMerger()

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(electronMergedSeeds, TkBasedSeeds = '')

