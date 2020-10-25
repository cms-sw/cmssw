import FWCore.ParameterSet.Config as cms


electronMergedSeeds =cms.EDProducer("ElectronSeedMerger",
     EcalBasedSeeds = cms.InputTag("ecalDrivenElectronSeeds"),
     TkBasedSeeds  = cms.InputTag("trackerDrivenElectronSeeds:SeedsForGsf")
    )

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toModify(electronMergedSeeds, TkBasedSeeds = '')

electronMergedSeedsFromMultiCl = electronMergedSeeds.clone(
  EcalBasedSeeds = 'ecalDrivenElectronSeedsFromMultiCl'
)
